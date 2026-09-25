"""
DraftKings NFL Showdown (Captain Mode) optimizer core: loading, options, run().

Nothing here prints, exits, or reads argparse; see cli/showdown.py for the
command line. Showdown files carry no player ID, so locks and excludes name
players by DataFrame row (index label), each with an optional slot:
SlotSelection(rows=(idx,), slot="CPT").

Showdown roster construction:
    - 6 total roster slots: 1 Captain (CPT) + 5 FLEX.
    - Any position (QB, RB, WR, TE, K, DST) is eligible for any slot.
    - The Captain scores 1.5x fantasy points and costs 1.5x salary.
    - A player may occupy the Captain slot OR a FLEX slot, never both.
    - The lineup must contain at least one player from each of the two teams.
    - Total salary must stay within the $50,000 cap.

Model: two binary vars per player (Captain, FLEX), built once and solved
repeatedly with a slot-aware diversity cut appended after each solve.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pulp

from nfl_dfs_optimizer import common
from nfl_dfs_optimizer.common import (
    OPTIMIZATION_TARGETS,
    SALARY_CAP,
    TARGET_PROJECTION,
    PlayerDataError,
    build_dk_upload_values,
    build_solver,
    check_target_data,
    target_score,
)

ROSTER_SIZE: int = 6
FLEX_SLOTS: int = ROSTER_SIZE - 1
CAPTAIN_MULTIPLIER: float = 1.5

# Projections CSV: the newest match in Downloads unless a path is given.
# Re-downloads ("... (1).csv") match too.
PROJECTIONS_GLOB: str = "DK NFL Showdown Projections*.csv"

# Maps the header names used by the Showdown projections export to the internal
# names used throughout. Order matters: when a file carries more than one alias
# for the same internal name (e.g. both "Total Own" and "Own"), the first one
# listed wins and the rest are left untouched, so the rename can never produce
# duplicate columns.
COLUMN_ALIASES: dict[str, str] = {
    "Pos": "Position",
    "Proj": "Projection",
    "Total Own": "Ownership",
    "Own": "Ownership",
    "CPT Own": "CptOwnership",
    "CPT Salary": "CptSalary",
    "CPT Proj": "CptProjection",
    "CPT Ceiling": "CptCeiling",
}

SLOT_CPT: str = "CPT"
SLOT_FLEX: str = "FLEX"
VALID_SLOTS: tuple[str, str] = (SLOT_CPT, SLOT_FLEX)

# Column order of the exported CSV. The DraftKings upload row reuses the
# columns after "Lineup_ID" as anonymous slots, one per rostered player.
EXPORT_COLUMNS: list[str] = [
    "Lineup_ID",
    "Slot",
    "Player",
    "Position",
    "Team",
    "Salary",
    "Projection",
    "Ownership",
    "Ceiling",
]


def target_value(player: Any, target: str, captain: bool) -> float:
    """
    Returns the value the solver maximizes for one player in one slot; the
    Captain slot uses the Captain columns (CptProjection / CptCeiling).
    """
    _, proj_weight, ceiling_weight = OPTIMIZATION_TARGETS[target]
    proj_col = "CptProjection" if captain else "Projection"
    ceiling_col = "CptCeiling" if captain else "Ceiling"
    return proj_weight * float(player[proj_col]) + ceiling_weight * float(player[ceiling_col])


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strips currency, percent, and whitespace characters, then coerces to numeric."""
    return (
        series.astype(str)
        .str.replace(r"[\$,%\s]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )


def _safe_name(text: Any) -> str:
    """Sanitizes arbitrary text into a token safe for use in a PuLP constraint name."""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(text))


# --- Loading ---


@dataclass
class PlayerPool:
    """A loaded Showdown projections file and what the load had to report."""

    df: pd.DataFrame
    notes: list[str]


def load_player_data(filepath: str) -> PlayerPool:
    """
    Loads and preprocesses Showdown player data from the projections CSV file,
    including the derived Captain-slot columns.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty; PlayerDataError (with the notes
            gathered so far) if critical columns are missing or the slate does
            not consist of exactly two teams.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Projections file not found at: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The projections file is empty: {filepath}")

    notes = [f"Successfully loaded {len(df)} players from {os.path.basename(filepath)}."]

    # Rename columns for consistency, applying only the aliases actually present
    # and never letting two of them collapse onto the same internal name.
    rename_map: dict[str, str] = {}
    claimed = set(df.columns)
    for source, target in COLUMN_ALIASES.items():
        if source not in df.columns or target in claimed:
            continue
        rename_map[source] = target
        claimed.add(target)
    df.rename(columns=rename_map, inplace=True)

    required_cols = ["Player", "Position", "Team", "Salary", "Projection"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise PlayerDataError(
            f"Projections file is missing required column(s): {', '.join(missing)}", notes
        )

    # Clean every numeric column that may arrive as "$10,600" or "67.4%".
    for col in [
        "Salary",
        "Projection",
        "Ceiling",
        "Ownership",
        "CptOwnership",
        "CptSalary",
        "CptProjection",
        "CptCeiling",
    ]:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    # Optional columns get sensible defaults so downstream math never breaks.
    if "Ceiling" not in df.columns:
        notes.append("  NOTE: No 'Ceiling' column found. Ceiling values will display as 0.00.")
        df["Ceiling"] = 0.0
    if "Ownership" not in df.columns:
        notes.append("  NOTE: No 'Total Own' column found. Ownership will display as 0.00%.")
        df["Ownership"] = 0.0
    if "CptOwnership" not in df.columns:
        notes.append("  NOTE: No 'CPT Own' column found. Captain ownership treated as 0.00%.")
        df["CptOwnership"] = 0.0

    df["Ownership"] = df["Ownership"].fillna(0.0)
    df["CptOwnership"] = df["CptOwnership"].fillna(0.0)

    # Drop players with missing critical data for optimization.
    df.dropna(subset=["Player", "Position", "Team", "Salary", "Projection"], inplace=True)
    df["Salary"] = df["Salary"].astype(int)

    # Ceiling is filled only after that drop, so the count below describes the
    # players actually available. A blank ceiling is not a reason to drop a
    # player, but it becomes a real zero, which the ceiling targets maximize.
    missing_ceiling = int(df["Ceiling"].isna().sum())
    if missing_ceiling:
        notes.append(
            f"  NOTE: {missing_ceiling} player(s) have no usable 'Ceiling' value; "
            f"treating it as 0.00."
        )
    df["Ceiling"] = df["Ceiling"].fillna(0.0)

    # --- Derived Captain-Slot Columns ---
    # Prefer the values supplied by the projections file; fall back to the
    # standard 1.5x Captain multiplier when a column is absent or blank. One
    # precedence for all three, so a source whose Captain values are not
    # exactly 1.5x scales the Captain identically under every target.
    if "CptSalary" in df.columns:
        df["CptSalary"] = df["CptSalary"].fillna(df["Salary"] * CAPTAIN_MULTIPLIER)
    else:
        df["CptSalary"] = df["Salary"] * CAPTAIN_MULTIPLIER
    df["CptSalary"] = df["CptSalary"].round().astype(int)

    if "CptProjection" in df.columns:
        df["CptProjection"] = df["CptProjection"].fillna(df["Projection"] * CAPTAIN_MULTIPLIER)
    else:
        df["CptProjection"] = df["Projection"] * CAPTAIN_MULTIPLIER

    if "CptCeiling" in df.columns:
        df["CptCeiling"] = df["CptCeiling"].fillna(df["Ceiling"] * CAPTAIN_MULTIPLIER)
    else:
        df["CptCeiling"] = df["Ceiling"] * CAPTAIN_MULTIPLIER

    # Slot-aware ownership: "Total Own" already includes "CPT Own", so a
    # player's FLEX-only ownership is the difference between the two.
    df["FlexOwnership"] = flex_ownership(df)

    # --- Slate Validation ---
    teams = sorted(df["Team"].astype(str).unique())
    if len(teams) != 2:
        raise PlayerDataError(
            f"Showdown slates must contain exactly two teams, but {len(teams)} were "
            f"found: {', '.join(teams)}. Filter the projections file down to a "
            f"single game before optimizing.",
            notes,
        )

    notes.append(
        f"Data preprocessed. {len(df)} players available for optimization "
        f"({teams[0]} vs {teams[1]})."
    )
    return PlayerPool(df, notes)


def flex_ownership(df: pd.DataFrame) -> pd.Series:
    """FLEX-only ownership: Total Own minus CPT Own, never below zero."""
    return (df["Ownership"] - df["CptOwnership"]).clip(lower=0.0)


def duplicate_player_groups(df: pd.DataFrame) -> dict[tuple[str, str], pd.Index]:
    """
    Players listed on more than one row (a duplicate projections export),
    keyed by (lowercased name, team). Each group may fill one roster spot.
    """
    return {
        key: idxs
        for key, idxs in df.groupby(
            [
                df["Player"].astype(str).str.strip().str.lower(),
                df["Team"].astype(str),
            ]
        ).groups.items()
        if len(idxs) > 1
    }


# --- Options and results ---


@dataclass(frozen=True)
class SlotSelection:
    """
    A lock or exclude: DataFrame rows and the slot it applies to ("CPT",
    "FLEX", or None for either). A lock over several rows (one player listed
    twice) puts exactly one of them in the slot.
    """

    rows: tuple[Any, ...]
    slot: str | None = None

    def __post_init__(self) -> None:
        if self.slot not in (None, *VALID_SLOTS):
            raise ValueError(f"Slot must be CPT, FLEX, or None, not {self.slot!r}.")
        if not self.rows:
            raise ValueError("A lock or exclude needs at least one player row.")


@dataclass(frozen=True)
class ShowdownOptions:
    """Every Showdown CLI flag except the projections path."""

    num_lineups: int = 1  # -n
    min_uniques: int = 1  # -u, counted by roster spot
    locks: tuple[SlotSelection, ...] = ()  # -l
    excludes: tuple[SlotSelection, ...] = ()  # -x
    max_salary: int = SALARY_CAP  # -ms; above the cap is clamped
    min_salary: int = 0  # -mns; 0 = no floor
    target: str = TARGET_PROJECTION  # -c / -pj
    export: bool = False  # -e; run() never writes, see export_rows()
    dk_entries: str | None = None  # -dk

    @property
    def effective_max_salary(self) -> int:
        return min(self.max_salary, SALARY_CAP)

    def validate(self, notes: list[str] | None = None) -> None:
        """
        Raises ValueError for options the optimizer refuses, with the CLI's
        wording. A --max-salary above the cap is not an error; it adds a
        clamping note.
        """
        if self.num_lineups < 1:
            raise ValueError("--num-lineups must be at least 1.")
        if not 1 <= self.min_uniques <= ROSTER_SIZE:
            raise ValueError(
                f"--min-uniques must be between 1 and {ROSTER_SIZE} (roster size)."
            )
        max_salary = self.effective_max_salary
        if self.max_salary > SALARY_CAP and notes is not None:
            notes.append(
                f"\nNOTE: --max-salary ${self.max_salary:,} exceeds the DraftKings "
                f"cap. Clamping to ${SALARY_CAP:,}."
            )
        if max_salary <= 0:
            raise ValueError("--max-salary must be a positive number.")
        if self.min_salary < 0:
            raise ValueError("--min-salary cannot be negative.")
        if self.min_salary > max_salary:
            raise ValueError(
                f"--min-salary ${self.min_salary:,} cannot exceed the lineup's "
                f"maximum salary of ${max_salary:,}."
            )
        if self.target not in OPTIMIZATION_TARGETS:
            raise ValueError(f"Unknown optimization target: {self.target!r}.")


@dataclass
class ShowdownLineup:
    """One solved lineup: CPT first, then the FLEX by descending salary."""

    number: int
    rows: list[dict[str, Any]]  # six slot-adjusted display rows
    captain_index: Any
    flex_indices: list[Any]
    projection: float
    ownership: float
    ceiling: float
    salary: int
    score: float


@dataclass
class ShowdownResult:
    """
    What run() produced. `status` is "Optimal" when every requested lineup was
    built, else the PuLP status of the solve that failed.
    """

    options: ShowdownOptions
    lineups: list[ShowdownLineup]
    status: str
    messages: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return len(self.lineups) >= self.options.num_lineups

    @property
    def lineups_df(self) -> pd.DataFrame:
        """One row per roster spot, lineups in order, CPT first."""
        return pd.DataFrame(
            [{"Lineup": lu.number, **row} for lu in self.lineups for row in lu.rows]
        )


def build_lineup_rows(
    players_df: pd.DataFrame, captain_idx: Any, flex_indices: list[Any]
) -> list[dict[str, Any]]:
    """
    Converts a solved lineup into ordered, slot-adjusted display rows.

    The Captain is listed first with its Captain salary, projection, ceiling,
    and ownership. The five FLEX players follow, ordered from highest to
    lowest salary.
    """
    captain = players_df.loc[captain_idx]
    rows: list[dict[str, Any]] = [
        {
            "Slot": SLOT_CPT,
            "Player": captain["Player"],
            "Position": captain["Position"],
            "Team": captain["Team"],
            "Salary": int(captain["CptSalary"]),
            "Projection": float(captain["CptProjection"]),
            "Ownership": float(captain["CptOwnership"]),
            "Ceiling": float(captain["CptCeiling"]),
        }
    ]

    flex_df = players_df.loc[list(flex_indices)].sort_values(
        by=["Salary", "Projection"], ascending=False
    )
    for _, player in flex_df.iterrows():
        rows.append(
            {
                "Slot": SLOT_FLEX,
                "Player": player["Player"],
                "Position": player["Position"],
                "Team": player["Team"],
                "Salary": int(player["Salary"]),
                "Projection": float(player["Projection"]),
                "Ownership": float(player["FlexOwnership"]),
                "Ceiling": float(player["Ceiling"]),
            }
        )
    return rows


def stop_messages(
    built: int, status: str, has_selectors: bool, max_salary: int, min_salary: int
) -> list[str]:
    """The lines explaining why a run stopped before building every lineup."""
    lines = [f"Could not find an optimal lineup. Status: {status}"]
    if built == 0:
        lines.append("This means no lineup exists that satisfies the constraints.")
        if has_selectors or max_salary < SALARY_CAP or min_salary > 0:
            lines.append(
                "  Check your --lock / --exclude selections and "
                "--max-salary / --min-salary; they are the usual cause."
            )
    else:
        lines.append(f"Stopped after generating {built} unique lineups.")
        if min_salary > 0:
            lines.append(
                f"  The ${min_salary:,} --min-salary floor shrinks "
                f"the pool; lowering it yields more lineups."
            )
    return lines


def _check_rows(df: pd.DataFrame, selections: tuple[SlotSelection, ...], action: str) -> None:
    for selection in selections:
        unknown = [row for row in selection.rows if row not in df.index]
        if unknown:
            raise ValueError(f"Cannot {action} player row(s) {unknown}: not in the projections.")


def run(players_df: pd.DataFrame, options: ShowdownOptions) -> ShowdownResult:
    """
    Builds up to options.num_lineups unique optimal Showdown lineups.

    Raises:
        ValueError: Invalid options, an unknown lock/exclude row, a ceiling
            target without ceiling data, or no HiGHS solver.
    """
    options.validate()
    messages: list[str] = []
    warning = check_target_data(players_df, options.target)
    if warning:
        messages.append(warning)
    _check_rows(players_df, options.locks, "lock")
    _check_rows(players_df, options.excludes, "exclude")

    df = players_df
    target = options.target
    max_salary = options.effective_max_salary
    players_dict = df.to_dict("index")
    player_indices = list(players_dict.keys())
    teams = sorted(df["Team"].astype(str).unique())

    prob = pulp.LpProblem("DraftKings_NFL_Showdown_Multi_Lineup", pulp.LpMaximize)
    # Two independent binary decisions per player: Captain and FLEX.
    cpt_vars = pulp.LpVariable.dicts("CPT", player_indices, cat="Binary")
    flex_vars = pulp.LpVariable.dicts("FLEX", player_indices, cat="Binary")

    # Evaluated per slot so the Captain contributes its Captain values under
    # any target.
    prob += (
        pulp.lpSum(
            target_value(players_dict[i], target, captain=True) * cpt_vars[i]
            + target_value(players_dict[i], target, captain=False) * flex_vars[i]
            for i in player_indices
        ),
        "Total_Target_Value",
    )

    # Salary cap (the clamped max) and the floor, bounding the same expression
    # from both sides so the Captain's salary counts once either way. A floor
    # of 0 adds no constraint at all.
    salary_expr = pulp.lpSum(
        players_dict[i]["CptSalary"] * cpt_vars[i] + players_dict[i]["Salary"] * flex_vars[i]
        for i in player_indices
    )
    prob += (salary_expr <= max_salary, "Salary_Cap")
    if options.min_salary > 0:
        prob += (salary_expr >= options.min_salary, "Min_Salary")
    prob += (pulp.lpSum(cpt_vars[i] for i in player_indices) == 1, "Captain_Slot")
    prob += (pulp.lpSum(flex_vars[i] for i in player_indices) == FLEX_SLOTS, "Flex_Slots")
    for i in player_indices:
        prob += (cpt_vars[i] + flex_vars[i] <= 1, f"One_Slot_Per_Player_{i}")
    # A player listed on more than one row must still fill at most one spot.
    for (name, team), idxs in duplicate_player_groups(df).items():
        prob += (
            pulp.lpSum(cpt_vars[i] + flex_vars[i] for i in idxs) <= 1,
            f"One_Row_Per_Player_{_safe_name(name)}_{_safe_name(team)}",
        )

    # Lineups must include at least one player from each team
    for team in teams:
        team_indices = df[df["Team"].astype(str) == team].index
        prob += (
            pulp.lpSum(cpt_vars[i] + flex_vars[i] for i in team_indices) >= 1,
            f"Min_One_From_{_safe_name(team)}",
        )

    for selection in dict.fromkeys(options.locks):
        rows, slot = selection.rows, selection.slot
        if slot == SLOT_CPT:
            expression = pulp.lpSum(cpt_vars[i] for i in rows)
        elif slot == SLOT_FLEX:
            expression = pulp.lpSum(flex_vars[i] for i in rows)
        else:
            expression = pulp.lpSum(cpt_vars[i] + flex_vars[i] for i in rows)
        tag = _safe_name(f"{'_'.join(str(r) for r in rows)}_{slot or 'ANY'}")
        prob += (expression == 1, f"Lock_{tag}")

    excluded: set[tuple[Any, str | None]] = set()
    for selection in options.excludes:
        for idx in selection.rows:
            if (idx, selection.slot) in excluded:
                continue
            excluded.add((idx, selection.slot))
            tag = _safe_name(f"{idx}_{selection.slot or 'ANY'}")
            if selection.slot == SLOT_CPT:
                prob += (cpt_vars[idx] == 0, f"Exclude_{tag}")
            elif selection.slot == SLOT_FLEX:
                prob += (flex_vars[idx] == 0, f"Exclude_{tag}")
            else:
                prob += (cpt_vars[idx] + flex_vars[idx] == 0, f"Exclude_{tag}")

    # --- Iterative Optimization Loop ---
    solver = build_solver()
    max_slots_can_share = ROSTER_SIZE - options.min_uniques
    lineups: list[ShowdownLineup] = []
    status = "Optimal"

    for i in range(options.num_lineups):
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            messages.extend(
                stop_messages(
                    i,
                    status,
                    bool(options.locks or options.excludes),
                    max_salary,
                    options.min_salary,
                )
            )
            break

        captain_idx = next(idx for idx in player_indices if (cpt_vars[idx].varValue or 0) > 0.5)
        flex_indices = [idx for idx in player_indices if (flex_vars[idx].varValue or 0) > 0.5]

        # Slot-aware diversity: this exact set of roster spots may not repeat,
        # so promoting a FLEX to Captain counts as two unique roster spots.
        prob += (
            cpt_vars[captain_idx] + pulp.lpSum(flex_vars[idx] for idx in flex_indices)
            <= max_slots_can_share,
            f"Diversity_from_lineup_{i + 1}",
        )

        rows = build_lineup_rows(df, captain_idx, flex_indices)
        projection = sum(r["Projection"] for r in rows)
        ceiling = sum(r["Ceiling"] for r in rows)
        lineups.append(
            ShowdownLineup(
                number=i + 1,
                rows=rows,
                captain_index=captain_idx,
                flex_indices=flex_indices,
                projection=projection,
                ownership=sum(r["Ownership"] for r in rows),
                ceiling=ceiling,
                salary=sum(r["Salary"] for r in rows),
                score=target_score(target, projection, ceiling),
            )
        )

    return ShowdownResult(options, lineups, status, messages)


# --- Export ---


def build_total_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The summary row that follows a lineup's six roster rows in the export."""
    return {
        "Slot": "TOTAL",
        "Player": "",
        "Position": "",
        "Team": "",
        "Salary": sum(r["Salary"] for r in rows),
        "Projection": sum(r["Projection"] for r in rows),
        "Ownership": sum(r["Ownership"] for r in rows),
        "Ceiling": sum(r["Ceiling"] for r in rows),
    }


def lineup_export_rows(
    lineup: ShowdownLineup,
    dk_lookup: dict[str, dict[str, Any]] | None,
    notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    One lineup's export rows: a row per roster spot, a TOTAL row, then the
    DraftKings upload row (omitted, with a note, when a player is unmatched).
    """
    out = [{**row, "Lineup_ID": lineup.number} for row in lineup.rows]
    out.append({**build_total_row(lineup.rows), "Lineup_ID": lineup.number})

    # DraftKings upload row: the same lineup laid out horizontally, one
    # "Name + ID" per roster slot in CPT-then-FLEX order.
    dk_values = build_dk_upload_values(lineup.rows, dk_lookup, notes)
    if dk_values:
        dk_row: dict[str, Any] = {"Lineup_ID": lineup.number}
        dk_row.update(zip(EXPORT_COLUMNS[1:], dk_values))
        out.append(dk_row)
    return out


def export_rows(
    result: ShowdownResult,
    dk_lookup: dict[str, dict[str, Any]] | None,
    notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Every lineup's export rows, in order."""
    return [row for lu in result.lineups for row in lineup_export_rows(lu, dk_lookup, notes)]


def write_export(rows: list[dict[str, Any]], target: str) -> str | None:
    """
    Writes export rows to EXPORT_DIR as
    nfl_showdown_multi_lineups[_ceiling|_projceiling]_<timestamp>.csv.

    Returns:
        The path written, or None when there was nothing to write.
    """
    if not rows:
        return None
    os.makedirs(common.EXPORT_DIR, exist_ok=True)
    path = common.export_path("nfl_showdown_multi_lineups", target)
    export_df = pd.DataFrame(rows).reindex(columns=EXPORT_COLUMNS)
    # Round derived floats so the export doesn't carry binary-float noise. The
    # upload rows put strings in these columns, so round per value.
    for col in ["Projection", "Ownership", "Ceiling"]:
        export_df[col] = export_df[col].map(lambda v: round(v, 2) if isinstance(v, float) else v)
    export_df.to_csv(path, index=False)
    return path

"""
DraftKings NFL Classic optimizer core: loading, options, and run().

Nothing here prints, exits, or reads argparse. The CLI (cli/classic.py) and
the Streamlit app both go through:

    pool = load_player_data(path)            # PlayerPool: df + load notes
    df = attach_kickoffs(pool.df, load_dk_kickoffs(entries_path))
    result = run(df, ClassicOptions(...))    # ClassicResult
    write_export(export_rows(result, lookup), slate, target)

Locks and excludes are DraftKings player IDs; matching names to IDs is the
caller's job.

Model: one binary var per player, built once and solved repeatedly, with a
diversity cut appended after each solve. Roster constraints are QB == 1,
RB >= 2, WR >= 3, TE >= 1, DST == 1, FLEX-eligible == 7, total == 9 -- the FLEX
slot is expressed as that count identity rather than a separate variable.
Which player prints as FLEX is decided afterward by _assign_flex_positions().
"""

import csv
import difflib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import pulp

from nfl_dfs_optimizer import common
from nfl_dfs_optimizer.common import (
    DK_ENTRIES_GLOB,
    OPTIMIZATION_TARGETS,
    SALARY_CAP,
    TARGET_PROJECTION,
    PlayerDataError,
    _find_dk_pool,
    build_dk_upload_values,
    build_solver,
    check_target_data,
    target_score,
)

ROSTER_SIZE: int = 9

# Display slots, in print and export order. The numbered slots print without
# their number ("RB1" -> "RB").
DISPLAY_ORDER: list[str] = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "FLEX", "DST"]

# Projections CSV: the newest match in Downloads unless a path is given. Main,
# Early, and Late slate downloads all match, re-downloads ("... (1).csv")
# included.
PROJECTIONS_GLOB: str = "DraftKings NFL DFS Projections*.csv"
# The slate is read from the file name ("... -- Main Slate.csv", "... - Early
# Slate.csv"). A name without one -- a bare "DraftKings NFL DFS
# Projections.csv", or any file passed by path -- is treated as Main.
SLATE_MAIN: str = "Main"
SLATE_PATTERN = re.compile(r"\b(main|early|late)\s+slate\b", re.IGNORECASE)
# Slate -> the tag inserted into the export file name after "nfl_classic".
# Main keeps the historical name.
SLATE_FILE_TAGS: dict[str, str] = {"Main": "", "Early": "_early", "Late": "_late"}

# Column order of the exported CSV. The DraftKings upload row reuses the
# columns after "Lineup_ID" as anonymous slots, one per rostered player.
EXPORT_COLUMNS: list[str] = [
    "Lineup_ID",
    "Player_ID",
    "Slot",
    "Player",
    "Position",
    "Team",
    "Salary",
    "Projection",
    "Ownership",
    "Ceiling",
]

# Game Info reads "GB@MIN 09/13/2026 04:25PM ET" before kickoff and
# "In Progress" after it. Kickoffs are Eastern wherever the code runs.
GAME_INFO_TIMEZONE: str = "America/New_York"
KICKOFF_PATTERN = re.compile(r"(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}[AP]M) ET")
KICKOFF_FORMAT: str = "%m/%d/%Y %I:%M%p"

# --- Projections CSV headers ---
# Internal column name -> the source headers that may carry it, in preference
# order. The projections source renamed several columns ("DK Pos", "DK Salary",
# "DK Proj", "DK Ceiling", "id"), so both generations of header are accepted.
# The first alias actually present wins, which keeps two source columns from
# ever collapsing onto one internal name.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ID": ("ID", "id", "DK ID", "Player ID"),
    "Player": ("Player", "Name", "Player Name"),
    "Position": ("Position", "DK Pos", "Pos"),
    "Team": ("Team", "Tm"),
    "Opp": ("Opp", "Opponent"),
    "Salary": ("Salary", "DK Salary"),
    "Projection": ("Projection", "Proj", "DK Proj"),
    "Ceiling": ("Ceiling", "DK Ceiling"),
}

# The file ships two ownership projections, and both are loaded:
# OwnershipLarge and OwnershipSmall. Which one fills the display-only
# "Ownership" column is ClassicOptions.ownership_field (the CLI's -sf).
#
# Only the large-field (default) column accepts the unlabeled legacy headers:
# "Own" does not say which field it measures, and on the files that carried it
# it was the only ownership column there was. Small field takes a column that
# actually says "Small Field" or nothing at all -- falling back to "Own" there
# would hand back large-field numbers under a request for small-field ones.
OWNERSHIP_LARGE_FIELD: str = "large"
OWNERSHIP_SMALL_FIELD: str = "small"
OWNERSHIP_FIELDS: tuple[str, str] = (OWNERSHIP_LARGE_FIELD, OWNERSHIP_SMALL_FIELD)

OWNERSHIP_ALIASES: dict[str, tuple[str, ...]] = {
    OWNERSHIP_LARGE_FIELD: ("Large Field", "Ownership", "Own"),
    OWNERSHIP_SMALL_FIELD: ("Small Field",),
}

OWNERSHIP_LABELS: dict[str, str] = {
    OWNERSHIP_LARGE_FIELD: "large field",
    OWNERSHIP_SMALL_FIELD: "small field",
}

# Where each field's ownership lives in the loaded DataFrame.
OWNERSHIP_COLUMNS: dict[str, str] = {
    OWNERSHIP_LARGE_FIELD: "OwnershipLarge",
    OWNERSHIP_SMALL_FIELD: "OwnershipSmall",
}

# Columns the optimizer cannot run without. "Ownership" and "Ceiling" stay
# optional and default to 0.00 when the file has neither name for them.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "ID",
    "Player",
    "Position",
    "Team",
    "Opp",
    "Salary",
    "Projection",
)

# Headers the fuzzy fallback must never claim. These are real columns with
# meanings of their own that sit one word away from a column we do want --
# "DK Proj" vs "DK Floor" vs "DK Ceiling" vs "DK Value" -- and a wrong guess
# among them would silently optimize on the wrong numbers.
UNMATCHABLE_HEADERS: tuple[str, ...] = ("DK Value", "Value", "DK Floor", "Floor")

# Minimum difflib similarity before an unrecognized header is accepted as a
# match. Deliberately high: erroring out is better than a silent mismatch.
FUZZY_HEADER_CUTOFF: float = 0.85

# Shortest target the fuzzy pass will match against. difflib's ratio is 2M/T
# over the combined length, so the cutoff gets weaker as the target gets
# shorter: against "own" any four-letter header containing that run -- "Down",
# "Town" -- scores 2*3/7 = 0.857 and clears 0.85. Six characters is where the
# ratio starts meaning what it says. Targets below it are simply not fuzzed,
# which costs nothing: a header close enough to "Tm" or "Opp" to be worth
# guessing at is already an exact alias hit.
MIN_FUZZY_TARGET_LENGTH: int = 6


def detect_slate(path: str) -> str:
    """Returns "Main", "Early", or "Late" from a projections file name; Main by default."""
    match = SLATE_PATTERN.search(os.path.basename(path))
    return match.group(1).title() if match else SLATE_MAIN


def target_value(player: dict[str, Any], target: str) -> float:
    """Returns the value the solver maximizes for one player under `target`."""
    _, proj_weight, ceiling_weight = OPTIMIZATION_TARGETS[target]
    return proj_weight * float(player["Projection"]) + ceiling_weight * float(
        player["Ceiling"]
    )


# --- Loading ---


def _normalize_header(header: Any) -> str:
    """Reduces a CSV header to a comparable token: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


def _alias_table(ownership_field: str) -> dict[str, tuple[str, ...]]:
    """Returns the full alias table with the requested ownership column folded in."""
    aliases = dict(COLUMN_ALIASES)
    aliases["Ownership"] = OWNERSHIP_ALIASES[ownership_field]
    return aliases


def resolve_columns(
    columns: list[Any],
    ownership_field: str = OWNERSHIP_LARGE_FIELD,
    notes: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Maps each internal column name onto a header actually present in the file.

    Two passes. The first takes exact alias hits, compared case- and
    punctuation-insensitively so "id", "ID", and "DK ID" are one header. The
    second is a difflib fuzzy fallback for anything still unresolved, so a
    future rename the alias table has not caught yet ("DK Projection") still
    lands. The fallback is deliberately narrow: it skips headers already
    claimed, headers that are a known alias of some *other* column, and the
    near-miss decoys in UNMATCHABLE_HEADERS; it ignores alias spellings shorter
    than MIN_FUZZY_TARGET_LENGTH, against which the ratio is too weak to mean
    anything; and it then demands FUZZY_HEADER_CUTOFF similarity. Every fuzzy
    hit is noted, because it is a guess.

    Returns:
        A (resolved, unresolved) pair: `resolved` maps internal column name to
        the source header supplying it; `unresolved` lists internal names with
        no header at all.
    """
    aliases = _alias_table(ownership_field)

    # First header wins a normalized spelling, so a file carrying both "Own"
    # and "own" does not produce a duplicate-column rename.
    by_normalized: dict[str, Any] = {}
    for column in columns:
        by_normalized.setdefault(_normalize_header(column), column)

    resolved: dict[str, Any] = {}
    claimed: set[str] = set()

    # Pass 1: exact alias hits, in the table's preference order.
    for internal, names in aliases.items():
        for name in names:
            header = by_normalized.get(_normalize_header(name))
            if header is None or _normalize_header(header) in claimed:
                continue
            resolved[internal] = header
            claimed.add(_normalize_header(header))
            break

    # Pass 2: fuzzy fallback. Reserve every alias of every column (both
    # ownership variants, so small field never silently eats "Large Field")
    # plus the decoy headers.
    reserved: set[str] = {
        _normalize_header(name)
        for table in (COLUMN_ALIASES, OWNERSHIP_ALIASES)
        for names in table.values()
        for name in names
    }
    reserved.update(_normalize_header(name) for name in UNMATCHABLE_HEADERS)

    for internal, names in aliases.items():
        if internal in resolved:
            continue
        # Targets are the alias list alone, never the internal name as well.
        # For every column but ownership the internal name *is* the first
        # alias, so this changes nothing; for ownership it is the point.
        # Adding "Ownership" unconditionally would let an unlabeled header
        # like "OwnershipPct" fuzzy-match for small field (2*9/21 = 0.857),
        # which is exactly the substitution small field promises never to
        # make. Built this way, the fuzzy pass honors the same
        # labeled/unlabeled rule the exact pass does.
        targets = {
            normalized
            for normalized in (_normalize_header(name) for name in names)
            if len(normalized) >= MIN_FUZZY_TARGET_LENGTH
        }
        if not targets:
            continue
        best_header: Any | None = None
        best_score = 0.0
        for column in columns:
            normalized = _normalize_header(column)
            if normalized in claimed or normalized in reserved:
                continue
            score = max(
                difflib.SequenceMatcher(None, normalized, target).ratio()
                for target in targets
            )
            if score > best_score:
                best_header, best_score = column, score
        if best_header is not None and best_score >= FUZZY_HEADER_CUTOFF:
            resolved[internal] = best_header
            claimed.add(_normalize_header(best_header))
            if notes is not None:
                notes.append(
                    f"  NOTE: Unrecognized header '{best_header}' matched to "
                    f"'{internal}' (similarity {best_score:.2f})."
                )

    unresolved = [internal for internal in aliases if internal not in resolved]
    return resolved, unresolved


@dataclass
class PlayerPool:
    """
    A loaded projections file.

    `df` carries both ownership projections (OWNERSHIP_COLUMNS) plus an
    "Ownership" column holding large field; run() refills it from
    ClassicOptions.ownership_field. `notes` is what the load had to report,
    per ownership field, since which column fed ownership changes the wording.
    """

    df: pd.DataFrame
    notes: dict[str, list[str]]


def _rename_plan(
    columns: list[Any], resolved: dict[str, Any], notes: list[str]
) -> tuple[dict[Any, str], list[Any]]:
    """The rename one ownership field's resolution implies, and what it supersedes."""
    rename_map = {
        header: internal for internal, header in resolved.items() if header != internal
    }
    # A header the resolver did not pick can still collide with a rename
    # target, which would leave two columns sharing one name. The live case is
    # small field against a file carrying both "Small Field" and a literal
    # "Ownership": the small-field request wins, so the unchosen column is
    # dropped rather than allowed to silently supply ownership.
    chosen = set(resolved.values())
    targets = set(rename_map.values())
    superseded = [column for column in columns if column in targets and column not in chosen]
    for column in superseded:
        notes.append(
            f"  NOTE: Ignoring the file's own '{column}' column; "
            f"'{resolved[column]}' supplies it instead."
        )
    for internal, header in sorted(resolved.items()):
        if header != internal:
            notes.append(f"  Mapped column '{header}' -> '{internal}'.")
    return rename_map, superseded


def _clean_ownership(values: Any, index: pd.Index) -> pd.Series:
    """Clean an ownership column ("11.70%" -> 11.70); a scalar fills every row."""
    series = values if isinstance(values, pd.Series) else pd.Series(values, index=index)
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def load_player_data(filepath: str) -> PlayerPool:
    """
    Loads and preprocesses player data from the projections CSV file.

    Headers are resolved through COLUMN_ALIASES rather than taken literally, so
    both the legacy names ("Proj", "Own", "ID") and the current ones
    ("DK Proj", "Large Field", "id") load without editing the file by hand.
    Both ownership projections are kept.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty; PlayerDataError if critical columns
            are missing (with the notes gathered so far).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Projections file not found at: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The projections file is empty: {filepath}")

    loaded = f"Successfully loaded {len(df)} players from {os.path.basename(filepath)}."
    notes: dict[str, list[str]] = {f: [loaded] for f in OWNERSHIP_FIELDS}

    # Resolve the headers once per ownership field: the fuzzy pass, the
    # superseded-column drop and the "Mapped" notes can all differ by field.
    source_headers = list(df.columns)
    resolutions = {
        f: resolve_columns(source_headers, f, notes[f]) for f in OWNERSHIP_FIELDS
    }

    for f in OWNERSHIP_FIELDS:
        missing_required = [n for n in REQUIRED_COLUMNS if n in resolutions[f][1]]
        if missing_required:
            raise PlayerDataError(
                "Projections file is missing required column(s): "
                f"{', '.join(missing_required)}.\n"
                f"  Headers found: {', '.join(str(h) for h in source_headers)}\n"
                "  Add the column to the file, or add its header to COLUMN_ALIASES.",
                notes,
            )

    ownership_sources = {
        f: (df[resolutions[f][0]["Ownership"]].copy() if "Ownership" in resolutions[f][0] else None)
        for f in OWNERSHIP_FIELDS
    }
    plans = {f: _rename_plan(source_headers, resolutions[f][0], notes[f]) for f in OWNERSHIP_FIELDS}

    # Every column but ownership resolves the same way under either field, so
    # the large-field plan shapes the frame; each field's ownership is taken
    # from its own source header.
    rename_map, superseded = plans[OWNERSHIP_LARGE_FIELD]
    if superseded:
        df.drop(columns=superseded, inplace=True)
    df.rename(columns=rename_map, inplace=True)

    # Clean Salary column (e.g., "$6,000 " -> 6000)
    df["Salary"] = (
        df["Salary"]
        .astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Ownership is display-and-export only -- no constraint or objective reads
    # it -- so a file without an ownership column still optimizes.
    for f in OWNERSHIP_FIELDS:
        source = ownership_sources[f]
        if source is None:
            notes[f].append(
                f"  NOTE: No {OWNERSHIP_LABELS[f]} ownership column found. "
                f"Ownership will display as 0.00%."
            )
            source = 0.0
        df[OWNERSHIP_COLUMNS[f]] = _clean_ownership(source, df.index)
    df["Ownership"] = df[OWNERSHIP_COLUMNS[OWNERSHIP_LARGE_FIELD]]

    def note_all(text: str) -> None:
        for f in OWNERSHIP_FIELDS:
            notes[f].append(text)

    # Ceiling drives the ceiling targets and the printed totals, but it is
    # optional: a file without it still optimizes on projection.
    if "Ceiling" not in df.columns:
        note_all("  NOTE: No 'Ceiling' column found. Ceiling values will display as 0.00.")
        df["Ceiling"] = 0.0
    df["Ceiling"] = (
        df["Ceiling"]
        .astype(str)
        .str.replace(r"[\$,%\s]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Projection may arrive as text, and a source that reformats its headers
    # may reformat its numbers too, so strip it like Salary and Ceiling. An
    # unparseable value becomes NaN and the row is dropped below.
    df["Projection"] = (
        df["Projection"]
        .astype(str)
        .str.replace(r"[\$,%\s]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # ID is coerced for the same reason: a non-integral id would otherwise
    # reach .astype(int) below and raise without naming the column or the row.
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")

    # Drop players with missing critical data for optimization
    critical_cols = ["ID", "Salary", "Projection", "Position"]
    df.dropna(subset=critical_cols, inplace=True)
    df["ID"] = df["ID"].astype(int)

    # Ceiling is filled only after that drop, so the count below describes the
    # players actually available. A blank ceiling is not a reason to drop a
    # player, but it becomes a real zero, which the ceiling targets maximize.
    missing_ceiling = int(df["Ceiling"].isna().sum())
    if missing_ceiling:
        note_all(
            f"  NOTE: {missing_ceiling} player(s) have no usable 'Ceiling' value; "
            f"treating it as 0.00."
        )
    df["Ceiling"] = df["Ceiling"].fillna(0.0)

    # --- Game Identification ---
    def get_game_id(row: pd.Series) -> frozenset[str]:
        """Creates a canonical, order-independent ID for a game."""
        team1 = str(row["Team"])
        # Opponent can be "@OPP" or "OPP", remove "@"
        team2 = str(row["Opp"]).replace("@", "")
        return frozenset([team1, team2])

    df["game_id"] = df.apply(get_game_id, axis=1)

    # --- Positional Flags for Constraints ---
    df["is_QB"] = (df["Position"] == "QB").astype(int)
    df["is_RB"] = (df["Position"] == "RB").astype(int)
    df["is_WR"] = (df["Position"] == "WR").astype(int)
    df["is_TE"] = (df["Position"] == "TE").astype(int)
    df["is_DST"] = (df["Position"] == "DST").astype(int)
    df["is_FLEX"] = df["Position"].isin(["RB", "WR", "TE"]).astype(int)

    note_all(f"Data preprocessed. {len(df)} players available for optimization.")
    return PlayerPool(df, notes)


def with_ownership(df: pd.DataFrame, ownership_field: str) -> pd.DataFrame:
    """A copy of `df` whose "Ownership" column holds the requested field."""
    if ownership_field not in OWNERSHIP_COLUMNS:
        raise ValueError(f"Unknown ownership field: {ownership_field!r}.")
    missing = [c for c in OWNERSHIP_COLUMNS.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"The players DataFrame has no {', '.join(missing)} column; "
            f"load it with classic.load_player_data()."
        )
    out = df.copy()
    out["Ownership"] = out[OWNERSHIP_COLUMNS[ownership_field]]
    return out


# --- Kickoffs (FLEX seating) ---


def _game_info_timezone() -> tzinfo:
    """Returns the Eastern zone Game Info kickoffs are written in."""
    try:
        return ZoneInfo(GAME_INFO_TIMEZONE)
    except ZoneInfoNotFoundError as exc:
        # Windows ships no zone database; Python reads it from the tzdata
        # package instead.
        raise ValueError(
            f"Time zone '{GAME_INFO_TIMEZONE}' is unavailable. The tzdata package "
            f"supplies it: run uv sync (or pip install tzdata in a plain venv)."
        ) from exc


def parse_kickoff(game_info: str, zone: tzinfo) -> datetime | None:
    """
    Reads the kickoff out of a Game Info cell ("GB@MIN 09/13/2026 04:25PM ET").

    Returns:
        The kickoff as an aware datetime, or None when the cell carries no
        start time -- "In Progress", "Final", or anything unrecognized.
    """
    match = KICKOFF_PATTERN.search(game_info)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), KICKOFF_FORMAT).replace(tzinfo=zone)
    except ValueError:
        return None


def load_dk_kickoffs(path: str | None, notes: list[str] | None = None) -> dict[int, datetime]:
    """
    Maps DraftKings player IDs to kickoffs, for seating the latest one in FLEX.

    Reads the player pool of the entries file. A lineup build does not need
    it, so every failure (no file, unreadable file, no Eastern zone) adds a
    note and returns an empty dict, which leaves the FLEX in salary order. A
    player whose Game Info shows no start time ("In Progress") is left out and
    so sorts as the earliest kickoff.
    """
    notes = notes if notes is not None else []
    skipped = "the FLEX is not reordered by kickoff."
    if path is None:
        notes.append(f"\nNOTE: No {DK_ENTRIES_GLOB} file in {common.DOWNLOADS_DIR}; {skipped}")
        return {}
    filename = os.path.basename(path)
    try:
        zone = _game_info_timezone()
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
    except (OSError, ValueError, csv.Error) as exc:
        notes.append(f"\nNOTE: Could not read kickoffs from {filename} ({exc}); {skipped}")
        return {}

    columns, pool_rows = _find_dk_pool(rows) or ({}, [])
    if "ID" not in columns or "Game Info" not in columns:
        notes.append(f"\nNOTE: {filename} has no player pool with ID and Game Info; {skipped}")
        return {}
    id_col, info_col = columns["ID"], columns["Game Info"]

    kickoffs: dict[int, datetime] = {}
    for cells in pool_rows:
        if len(cells) <= max(id_col, info_col):
            continue
        try:
            dk_id = int(cells[id_col].strip())
        except ValueError:
            continue
        kickoff = parse_kickoff(cells[info_col], zone)
        if kickoff is not None:
            kickoffs[dk_id] = kickoff
    if kickoffs:
        notes.append(f"\nKickoff times read from {filename} ({len(kickoffs)} players).")
    else:
        notes.append(f"\nNOTE: {filename} lists no readable kickoff times; {skipped}")
    return kickoffs


def attach_kickoffs(
    df: pd.DataFrame, kickoffs: dict[int, datetime], notes: list[str] | None = None
) -> pd.DataFrame:
    """
    A copy of `df` with a "Kickoff" column mapped from `kickoffs` by player ID.

    Kickoffs only reorder the display slots (the latest one sits in FLEX); they
    never touch the model. Matched by DraftKings ID, so a stale entries file
    from another slate simply matches no one.
    """
    out = df.copy()
    out["Kickoff"] = out["ID"].map(kickoffs)
    if kickoffs and notes is not None:
        notes.append(
            f"Kickoffs matched for {int(out['Kickoff'].notna().sum())} "
            f"of {len(out)} projected players."
        )
    return out


def _kickoff_timestamp(kickoff: Any) -> float:
    """Sort key for a kickoff; a missing one (unmatched, already started) sorts first."""
    return kickoff.timestamp() if pd.notna(kickoff) else float("-inf")


def format_kickoff(kickoff: Any) -> str:
    """A kickoff as its Eastern clock time ("4:25PM"), or "-" when unknown."""
    return f"{kickoff:%I:%M%p}".lstrip("0") if pd.notna(kickoff) else "-"


def _assign_flex_positions(lineup: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Assigns players from an optimal lineup to specific roster slots.

    Fills the primary RB, WR, and TE slots first, sorted by salary descending.
    The single remaining FLEX-eligible player is then placed in the FLEX slot.
    When the lineup carries a "Kickoff" column, the FLEX is the latest kickoff
    among the position that supplies it; the position counts fix which
    position that is, so a lineup with a third RB always has an RB in the FLEX.

    Returns:
        A dictionary mapping roster slots to player data dictionaries.
    """
    assigned_lineup: dict[str, dict[str, Any]] = {}
    unassigned_players = lineup.copy()

    # 1. Assign dedicated slots first (QB, DST)
    for pos in ["QB", "DST"]:
        player = unassigned_players[unassigned_players["Position"] == pos].iloc[0]
        assigned_lineup[pos] = player.to_dict()
        unassigned_players.drop(player.name, inplace=True)

    # 2. Assign RBs, WRs, TEs to their primary slots, sorted by salary descending
    for pos, count in [("RB", 2), ("WR", 3), ("TE", 1)]:
        positional_players = unassigned_players[
            unassigned_players["Position"] == pos
        ].sort_values(by="Salary", ascending=False)

        # The position with a surplus supplies the FLEX. Hold back its latest
        # kickoff for that slot -- a FLEX can be swapped for any RB/WR/TE, so
        # the latest-locking player keeps the most late-swap options open. On
        # a tie, or with no kickoff times at all, the last (cheapest) player
        # is held back, which is the salary order this function always used.
        if len(positional_players) > count and "Kickoff" in positional_players:
            order = [_kickoff_timestamp(k) for k in positional_players["Kickoff"]]
            flex_pos = max(range(len(order)), key=lambda k: (order[k], k))
            positional_players = positional_players.drop(positional_players.index[flex_pos])

        players_to_assign_to_slots = positional_players.head(count)
        for i, (idx, player) in enumerate(players_to_assign_to_slots.iterrows()):
            slot = f"{pos}{i + 1}"
            assigned_lineup[slot] = player.to_dict()
            unassigned_players.drop(idx, inplace=True)

    # 3. The single remaining player is the FLEX
    if not unassigned_players.empty:
        flex_player = unassigned_players.iloc[0]
        assigned_lineup["FLEX"] = flex_player.to_dict()

    return assigned_lineup


def display_slot(slot: str) -> str:
    """"RB1" -> "RB": the slot label printed and shown for a display slot."""
    return re.sub(r"\d", "", slot)


# --- Options and results ---


@dataclass(frozen=True)
class ClassicOptions:
    """
    Every Classic CLI flag except the projections path and -ls (late swap is
    CLI-only). Locks and excludes are DraftKings player IDs.

    `export` and `dk_entries` mirror -e and -dk for the caller, which does the
    exporting and picks the entries file; run() reads neither.
    """

    num_lineups: int = 1  # -n
    min_uniques: int = 1  # -u
    lock_ids: tuple[int, ...] = ()  # -l
    exclude_ids: tuple[int, ...] = ()  # -x
    stack: int = 0  # -s N; 0 = off
    stack_rb: bool = False  # -srb
    max_te: int | None = None  # -te
    no_dst_opp: bool = False  # -ndo
    min_salary: int = 0  # -mns; 0 = no floor
    target: str = TARGET_PROJECTION  # -c / -pj
    ownership_field: str = OWNERSHIP_LARGE_FIELD  # -sf
    export: bool = False  # -e; run() never writes, see export_rows()
    dk_entries: str | None = None  # -dk

    def validate(self, strict: bool = True) -> None:
        """
        Raises ValueError for options the optimizer refuses, with the CLI's
        wording.

        strict (the default) also range-checks the counts: num_lineups >= 1,
        1 <= min_uniques <= 9, stack >= 0, max_te >= 0. The CLI passes
        strict=False, keeping the checks the legacy script made -- under it a
        -u 0 still returns repeats and -n 0 builds nothing, as they always did.
        """
        if strict:
            if self.num_lineups < 1:
                raise ValueError("--num-lineups must be at least 1.")
            if not 1 <= self.min_uniques <= ROSTER_SIZE:
                raise ValueError(
                    f"--min-uniques must be between 1 and {ROSTER_SIZE} (roster size)."
                )
            if self.stack < 0:
                raise ValueError("--stack cannot be negative.")
            if self.max_te is not None and self.max_te < 0:
                raise ValueError("--max-te cannot be negative.")
        if self.min_salary < 0:
            raise ValueError("--min-salary cannot be negative.")
        if self.min_salary > SALARY_CAP:
            raise ValueError(
                f"--min-salary ${self.min_salary:,} exceeds the "
                f"${SALARY_CAP:,} DraftKings salary cap."
            )
        if self.target not in OPTIMIZATION_TARGETS:
            raise ValueError(f"Unknown optimization target: {self.target!r}.")
        if self.ownership_field not in OWNERSHIP_COLUMNS:
            raise ValueError(f"Unknown ownership field: {self.ownership_field!r}.")


@dataclass
class ClassicLineup:
    """One solved lineup, seated into display slots."""

    number: int
    players: dict[str, dict[str, Any]]  # display slot ("RB1") -> player row
    projection: float
    ownership: float
    ceiling: float
    salary: int
    score: float  # the objective value under the run's target

    def slot_rows(self) -> list[tuple[str, dict[str, Any] | None]]:
        """(display slot, player row or None) in DISPLAY_ORDER."""
        return [(slot, self.players.get(slot)) for slot in DISPLAY_ORDER]


@dataclass
class ClassicResult:
    """
    What run() produced.

    `status` is "Optimal" when every requested lineup was built, else the
    PuLP status of the solve that failed. `messages` explains a run that
    stopped short (the lines the CLI prints); `warning` is the partial-ceiling
    warning from check_target_data(), if any.
    """

    options: ClassicOptions
    lineups: list[ClassicLineup]
    status: str
    messages: list[str] = field(default_factory=list)
    show_kickoff: bool = False
    warning: str | None = None

    @property
    def complete(self) -> bool:
        return len(self.lineups) >= self.options.num_lineups

    @property
    def lineups_df(self) -> pd.DataFrame:
        """One row per rostered player, lineups in order, slots in display order."""
        rows = []
        for lineup in self.lineups:
            for slot, player in lineup.slot_rows():
                if player is None:
                    continue
                rows.append(
                    {
                        "Lineup": lineup.number,
                        "Slot": display_slot(slot),
                        "Player_ID": player["ID"],
                        "Player": player["Player"],
                        "Position": player["Position"],
                        "Team": player["Team"],
                        "Opp": player["Opp"],
                        "Salary": int(player["Salary"]),
                        "Projection": player["Projection"],
                        "Ownership": player["Ownership"],
                        "Ceiling": player["Ceiling"],
                        "Kickoff": player.get("Kickoff"),
                    }
                )
        return pd.DataFrame(rows)


def stop_messages(built: int, status: str, min_salary: int) -> list[str]:
    """The lines explaining why a run stopped before building every lineup."""
    lines = [f"Could not find an optimal lineup. Status: {status}"]
    if built == 0:
        lines.append("This means no lineup exists that satisfies the base constraints.")
        if min_salary > 0:
            lines.append(
                f"  A ${min_salary:,} --min-salary floor is a common cause; try lowering it."
            )
    else:
        lines.append(f"Stopped after generating {built} unique lineups.")
        if min_salary > 0:
            lines.append(
                f"  The ${min_salary:,} --min-salary floor shrinks the pool; "
                f"lowering it yields more lineups."
            )
    return lines


def _rows_for_ids(df: pd.DataFrame, ids: tuple[int, ...], action: str) -> list[Any]:
    """DataFrame index labels for player IDs, in order; unknown IDs raise."""
    rows: list[Any] = []
    for dk_id in dict.fromkeys(ids):
        matches = df.index[df["ID"] == dk_id]
        if matches.empty:
            raise ValueError(f"Cannot {action} player ID {dk_id}: not in the projections.")
        rows.extend(idx for idx in matches if idx not in rows)
    return rows


def run(
    players_df: pd.DataFrame, options: ClassicOptions, *, strict: bool = True
) -> ClassicResult:
    """
    Builds up to options.num_lineups unique optimal lineups.

    `players_df` comes from load_player_data(), optionally through
    attach_kickoffs() for kickoff-based FLEX seating. `strict` is passed to
    options.validate(); only the CLI turns it off.

    Raises:
        ValueError: Invalid options, an unknown lock/exclude ID, a ceiling
            target without ceiling data, or no HiGHS solver.
    """
    options.validate(strict)
    messages: list[str] = []
    warning = check_target_data(players_df, options.target)

    df = with_ownership(players_df, options.ownership_field)
    lock_rows = _rows_for_ids(df, options.lock_ids, "lock")
    exclude_rows = _rows_for_ids(df, options.exclude_ids, "exclude")
    show_kickoff = "Kickoff" in df and bool(df["Kickoff"].notna().any())

    players_dict = df.to_dict("index")
    player_indices = list(players_dict.keys())
    unique_game_ids = list(df["game_id"].unique())

    # --- Define the Optimization Problem (once) ---
    prob = pulp.LpProblem("DraftKings_NFL_Multi_Lineup", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Player", player_indices, cat="Binary")
    game_vars = pulp.LpVariable.dicts("Game", unique_game_ids, cat="Binary")

    # The objective is a weighted mix of projection and ceiling; the weights
    # come from the chosen target and never change mid-run.
    prob += (
        pulp.lpSum(
            target_value(players_dict[i], options.target) * player_vars[i]
            for i in player_indices
        ),
        "Total_Target_Value",
    )

    # Salary Cap, and the floor on the same quantity. A floor of 0 adds no
    # constraint at all.
    salary_expr = pulp.lpSum(players_dict[i]["Salary"] * player_vars[i] for i in player_indices)
    prob += (salary_expr <= SALARY_CAP, "Salary_Cap")
    if options.min_salary > 0:
        prob += (salary_expr >= options.min_salary, "Min_Salary")
    prob += (
        pulp.lpSum(player_vars[i] for i in player_indices) == ROSTER_SIZE,
        "Total_Players",
    )

    def position_sum(flag: str) -> Any:
        return pulp.lpSum(players_dict[i][flag] * player_vars[i] for i in player_indices)

    prob += (position_sum("is_QB") == 1, "QB_Slot")
    prob += (position_sum("is_RB") >= 2, "Min_RB")
    prob += (position_sum("is_WR") >= 3, "Min_WR")
    prob += (position_sum("is_TE") >= 1, "Min_TE")
    prob += (position_sum("is_DST") == 1, "DST_Slot")
    prob += (position_sum("is_FLEX") == 7, "FLEX_Logic")

    # Game Diversity Rule: at least two games.
    for p_idx in player_indices:
        game_id = players_dict[p_idx]["game_id"]
        prob += (
            game_vars[game_id] >= player_vars[p_idx],
            f"Link_Player_{p_idx}_to_Game",
        )
    prob += (
        pulp.lpSum(game_vars[gid] for gid in unique_game_ids) >= 2,
        "At_Least_Two_Games",
    )

    for idx in lock_rows:
        prob += player_vars[idx] == 1, f"Lock_{idx}"
    for idx in exclude_rows:
        prob += player_vars[idx] == 0, f"Exclude_{idx}"

    if options.stack > 0:
        for qb_idx, qb_row in df[df["Position"] == "QB"].iterrows():
            team = qb_row["Team"]
            partners = df[(df["Team"] == team) & (df["Position"].isin(["WR", "TE"]))].index
            prob += (
                pulp.lpSum(player_vars[i] for i in partners)
                >= options.stack * player_vars[qb_idx],
                f"Stack_QB_{qb_idx}_{team}_WRTE",
            )

    if options.stack_rb:
        for qb_idx, qb_row in df[df["Position"] == "QB"].iterrows():
            team = qb_row["Team"]
            partners = df[(df["Team"] == team) & (df["Position"] == "RB")].index
            prob += (
                pulp.lpSum(player_vars[i] for i in partners) >= player_vars[qb_idx],
                f"Stack_QB_{qb_idx}_{team}_RB",
            )

    if options.no_dst_opp:
        for dst_idx, dst_row in df[df["Position"] == "DST"].iterrows():
            opp_team = str(dst_row["Opp"]).replace("@", "")
            opp_offense = df[
                (df["Team"] == opp_team) & (df["Position"].isin(["QB", "RB", "WR", "TE"]))
            ].index
            for off_idx in opp_offense:
                prob += (
                    player_vars[dst_idx] + player_vars[off_idx] <= 1,
                    f"No_DST_{dst_idx}_vs_Opp_{off_idx}",
                )

    if options.max_te is not None:
        prob += (position_sum("is_TE") <= options.max_te, "Max_TE_Constraint")

    # --- Iterative Optimization Loop ---
    solver = build_solver()
    max_players_can_share = ROSTER_SIZE - options.min_uniques
    lineups: list[ClassicLineup] = []
    status = "Optimal"

    for i in range(options.num_lineups):
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            messages.extend(stop_messages(i, status, options.min_salary))
            break

        selected = [p for p in player_indices if player_vars[p].varValue > 0.5]
        # This exact lineup may not share more than the allowance again.
        prob += (
            pulp.lpSum(player_vars[p] for p in selected) <= max_players_can_share,
            f"Diversity_from_lineup_{i + 1}",
        )

        lineup_df = df.loc[selected].copy()
        projection = lineup_df["Projection"].sum()
        ceiling = lineup_df["Ceiling"].sum()
        lineups.append(
            ClassicLineup(
                number=i + 1,
                players=_assign_flex_positions(lineup_df),
                projection=projection,
                ownership=lineup_df["Ownership"].sum(),
                ceiling=ceiling,
                salary=int(lineup_df["Salary"].sum()),
                score=target_score(options.target, projection, ceiling),
            )
        )

    return ClassicResult(options, lineups, status, messages, show_kickoff, warning)


# --- Export ---


def lineup_export_rows(
    lineup: ClassicLineup,
    dk_lookup: dict[str, dict[str, Any]] | None,
    notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    One lineup's export rows: a row per roster spot, a TOTAL row, then the
    DraftKings upload row of "Name + ID" values (omitted, with a note, when
    the entries file cannot supply every player).
    """
    player_rows = []
    for slot, player in lineup.slot_rows():
        if player:
            row = player.copy()
            row["Lineup_ID"] = lineup.number
            row["Slot"] = slot
            row["Player_ID"] = row.get("ID")
            player_rows.append(row)
    rows = list(player_rows)

    # Summary row for the lineup, matching the printed totals.
    rows.append(
        {
            "Lineup_ID": lineup.number,
            "Player_ID": "",
            "Slot": "TOTAL",
            "Player": "",
            "Position": "",
            "Team": "",
            "Salary": lineup.salary,
            "Projection": round(lineup.projection, 2),
            "Ownership": round(lineup.ownership, 2),
            "Ceiling": round(lineup.ceiling, 2),
        }
    )

    # DraftKings upload row: the same lineup laid out horizontally, one
    # "Name + ID" per roster slot in display order.
    dk_values = build_dk_upload_values(player_rows, dk_lookup, notes)
    if dk_values:
        dk_row: dict[str, Any] = {"Lineup_ID": lineup.number}
        dk_row.update(zip(EXPORT_COLUMNS[1:], dk_values))
        rows.append(dk_row)
    return rows


def export_rows(
    result: ClassicResult,
    dk_lookup: dict[str, dict[str, Any]] | None,
    notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Every lineup's export rows, in order."""
    return [row for lu in result.lineups for row in lineup_export_rows(lu, dk_lookup, notes)]


def write_export(rows: list[dict[str, Any]], slate: str, target: str) -> str | None:
    """
    Writes export rows to EXPORT_DIR as
    nfl_classic[_early|_late]_multi_lineups[_ceiling|_projceiling]_<timestamp>.csv.

    Returns:
        The path written, or None when there was nothing to write.
    """
    if not rows:
        return None
    os.makedirs(common.EXPORT_DIR, exist_ok=True)
    path = common.export_path(f"nfl_classic{SLATE_FILE_TAGS[slate]}_multi_lineups", target)
    pd.DataFrame(rows).reindex(columns=EXPORT_COLUMNS).to_csv(path, index=False)
    return path

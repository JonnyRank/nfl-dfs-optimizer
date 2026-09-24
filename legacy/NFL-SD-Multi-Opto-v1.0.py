"""
DraftKings NFL Showdown (Captain Mode) Multi-Lineup Optimizer.

This script ingests a CSV file with Showdown player projections and uses mixed
integer linear programming (PuLP + HiGHS) to find a specified number of unique,
optimal lineups that maximize a chosen scoring target (projection, ceiling, or a
50/50 mix of the two), subject to DraftKings' Showdown contest rules.

Showdown roster construction:
    - 6 total roster slots: 1 Captain (CPT) + 5 FLEX.
    - Any position (QB, RB, WR, TE, K, DST) is eligible for any slot.
    - The Captain scores 1.5x fantasy points and costs 1.5x salary.
    - A player may occupy the Captain slot OR a FLEX slot, never both.
    - The lineup must contain at least one player from each of the two teams.
    - Total salary must stay within the $50,000 cap.

The script is run from the command line. It optimizes on the newest
"DK NFL Showdown Projections*.csv" in your Downloads folder, or on the
projections CSV whose path is given as the first argument (a file, not a
folder).

Input Arguments:
    python NFL-SD-Multi-Opto-v1.0.py ["path"] -n -u -e -l -x -ms -mns -c -pj -dk
    python <script> <proj file (optional)> <# of lineups> <min uniques> <export to CSV> <lock players> <exclude players> <max salary> <minimum salary> <optimize on ceiling> <optimize on 50/50 proj+ceiling> <DKEntries file path>
    # Means: python <script> <projections file> -n <number of lineups> -u <min uniques> -e <export to CSV>
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -e -l "Drake Maye:CPT" -ms 49800
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -ms 49800 -mns 49000
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -c
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -pj
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -e -dk "C:\\path\\to\\DKEntries.csv"
    python NFL-SD-Multi-Opto-v1.0.py -n 5 -u 2 -e

DraftKings Entries File (-dk / --dk-entries):
    With -e, each lineup's export ends with an upload row of DraftKings
    "Name + ID" values read from the newest DKEntries*.csv in your Downloads
    folder; -dk names another file, and a -dk path that does not exist stops
    the run. No entries file just drops the upload rows.

Optimization Targets:
    (default)               Maximize total projection.
    -c / --ceiling          Maximize total ceiling.
    -pj / --projceiling     Maximize an equally weighted 50/50 blend of the two.
    Captain values are used for the Captain slot under every target: the
    file's "CPT Proj" / "CPT Ceiling" when present, otherwise 1.5x the FLEX
    value. The two flags are mutually exclusive; omitting both keeps the
    projection-only behavior.

Key Features:
- Loads Showdown player data (CPT salary / CPT projection / CPT ceiling / CPT
  ownership) from a command-line specified CSV file.
- Cleans and validates player salary, projection, and ownership data.
- Models the Captain and FLEX slots as separate binary decisions per player.
- Uses the HiGHS solver (via highspy) through PuLP to solve each lineup.
- Optimizes on projection, ceiling, or a 50/50 blend of the two.
- Enforces constraints for salary cap, max and min salary, roster composition,
  both-teams representation, and lineup diversity.
- Slot-aware ownership: the Captain contributes its CPT ownership and each FLEX
  contributes (Total Own - CPT Own), so the printed total is true product
  ownership for the exact lineup built.
- Prints a well-formatted, human-readable lineup as CPT + 5 FLEX, with the FLEX
  players ordered from highest to lowest salary.
"""

import os
import re
import csv
import glob
import argparse
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pulp

# --- Constants ---
SALARY_CAP: int = 50000
ROSTER_SIZE: int = 6
FLEX_SLOTS: int = ROSTER_SIZE - 1
CAPTAIN_MULTIPLIER: float = 1.5
EXPORT_DIR: str = r"G:\My Drive\Documents\NFL-DFS\csv-exports"
# John Rankin's (Dad) Downloads folder path: r"C:\Users\jdr0824\Downloads"

# Maps the header names used by the Showdown projections export to the internal
# names used throughout this script.
# Order matters: when a file carries more than one alias for the same internal
# name (e.g. both "Total Own" and "Own"), the first one listed wins and the
# rest are left untouched, so the rename can never produce duplicate columns.
COLUMN_ALIASES: Dict[str, str] = {
    "Pos": "Position",
    "Proj": "Projection",
    "Total Own": "Ownership",
    "Own": "Ownership",
    "CPT Own": "CptOwnership",
    "CPT Salary": "CptSalary",
    "CPT Proj": "CptProjection",
    "CPT Ceiling": "CptCeiling",
}

# Valid slot qualifiers for the -l / -x arguments, e.g. "Drake Maye:CPT".
SLOT_CPT: str = "CPT"
SLOT_FLEX: str = "FLEX"
VALID_SLOTS: Tuple[str, str] = (SLOT_CPT, SLOT_FLEX)

TABLE_WIDTH: int = 91

# --- Optimization Targets ---
# Each target weights a player's projection and ceiling into the single value
# the solver maximizes. The Captain slot uses the Captain-specific columns
# (CptProjection / CptCeiling), so the 1.5x multiplier carries through every
# target. Projection-only is the default and reproduces the original behavior.
TARGET_PROJECTION: str = "projection"
TARGET_CEILING: str = "ceiling"
TARGET_BLEND: str = "blend"

# target -> (label for output, projection weight, ceiling weight)
OPTIMIZATION_TARGETS: Dict[str, Tuple[str, float, float]] = {
    TARGET_PROJECTION: ("Projection", 1.0, 0.0),
    TARGET_CEILING: ("Ceiling", 0.0, 1.0),
    TARGET_BLEND: ("50/50 Projection + Ceiling", 0.5, 0.5),
}

# Suffix added to the export filename so a run's target is obvious on disk.
TARGET_FILE_SUFFIXES: Dict[str, str] = {
    TARGET_PROJECTION: "",
    TARGET_CEILING: "_ceiling",
    TARGET_BLEND: "_projceiling",
}

# DraftKings entries file used to translate rostered players into the
# "Name + ID" values DraftKings expects when a lineup is uploaded. -dk /
# --dk-entries names one explicitly; otherwise the newest match in the current
# user's Downloads folder wins, so a browser re-download ("DKEntries (1).csv")
# is picked up.
DOWNLOADS_DIR: str = os.path.join(os.path.expanduser("~"), "Downloads")
DK_ENTRIES_GLOB: str = "DKEntries*.csv"

# Projections CSV: the newest match in Downloads unless the filepath argument
# names one. Re-downloads ("... (1).csv") match too.
PROJECTIONS_GLOB: str = "DK NFL Showdown Projections*.csv"

# Name suffixes dropped when a projections name and a DraftKings name disagree.
NAME_SUFFIXES: Tuple[str, ...] = ("jr", "sr", "ii", "iii", "iv", "v")

# Column order of the exported CSV. The DraftKings upload row reuses the
# columns after "Lineup_ID" as anonymous slots, one per rostered player.
EXPORT_COLUMNS: List[str] = [
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


def resolve_optimization_target(use_ceiling: bool, use_blend: bool) -> str:
    """
    Turns the --ceiling / --projceiling flags into a single target key.

    Args:
        use_ceiling: True when -c / --ceiling was passed.
        use_blend: True when -pj / --projceiling was passed.

    Returns:
        One of TARGET_CEILING, TARGET_BLEND, or TARGET_PROJECTION (the default).

    Raises:
        ValueError: If both flags are supplied together. argparse's mutually
            exclusive group rejects that first, so a user never reaches this;
            it keeps the helper correct when called outside main().
    """
    if use_ceiling and use_blend:
        raise ValueError(
            "Choose only one optimization target: --ceiling or --projceiling."
        )
    if use_ceiling:
        return TARGET_CEILING
    if use_blend:
        return TARGET_BLEND
    return TARGET_PROJECTION


def target_value(player: Any, target: str, captain: bool) -> float:
    """
    Returns the value the solver maximizes for one player in one slot.

    Args:
        player: The player's row, as a mapping of column name to value.
        target: The active optimization target key.
        captain: True for the Captain slot, which uses the 1.5x columns.

    Returns:
        The weighted projection/ceiling value for that player-slot pair.
    """
    _, proj_weight, ceiling_weight = OPTIMIZATION_TARGETS[target]
    proj_col = "CptProjection" if captain else "Projection"
    ceiling_col = "CptCeiling" if captain else "Ceiling"
    return proj_weight * float(player[proj_col]) + ceiling_weight * float(
        player[ceiling_col]
    )


def validate_target_data(df: pd.DataFrame, target: str) -> None:
    """
    Checks the ceiling data a ceiling-weighted target is about to maximize.

    Ceiling is optional in the projections file and defaults to zero, so an
    all-zero column is fatal: it would silently return an arbitrary
    salary-feasible lineup. A partly-populated column is legal but
    consequential -- a zero ceiling is indistinguishable from a blank one once
    filled, and those players score nothing on the ceiling half of the
    objective -- so name them rather than letting the pool shrink invisibly.

    Raises:
        ValueError: If the target weights ceiling but no positive ceiling exists.
    """
    label, _, ceiling_weight = OPTIMIZATION_TARGETS[target]
    if ceiling_weight == 0:
        return
    if "Ceiling" not in df.columns or not (df["Ceiling"] > 0).any():
        raise ValueError(
            f"The '{label}' target needs a populated 'Ceiling' column, but the "
            f"projections file has no ceiling values. Re-run without "
            f"--ceiling/--projceiling to optimize on projection."
        )

    zeroed = df[df["Ceiling"] <= 0]
    if zeroed.empty:
        return
    consequence = (
        "they cannot be rostered unless locked"
        if target == TARGET_CEILING
        else "they are scored on their projection alone"
    )
    names = ", ".join(str(name) for name in zeroed["Player"].head(5))
    if len(zeroed) > 5:
        names += f", +{len(zeroed) - 5} more"
    print(
        f"  WARNING: {len(zeroed)} of {len(df)} players have a zero or missing "
        f"ceiling. Under the '{label}' target {consequence}: {names}"
    )


def _normalize_name(name: Any) -> str:
    """Lowercases a name and drops punctuation so it can be matched across files."""
    text = re.sub(r"[^a-z0-9 ]", "", str(name).lower())
    return re.sub(r"\s+", " ", text).strip()


def _strip_name_suffix(normalized: str) -> str:
    """Removes a trailing generational suffix (Jr., III, ...) from a normalized name."""
    parts = normalized.split()
    while len(parts) > 2 and parts[-1] in NAME_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _newest_download(pattern: str, directory: Optional[str] = None) -> Optional[str]:
    """
    Returns the most recently modified file in `directory` (default
    DOWNLOADS_DIR) matching `pattern`, or None when nothing matches. The
    newest wins, so a browser re-download ("... (1).csv") is picked up.
    """
    directory = directory or DOWNLOADS_DIR
    # Escape the folder so a "[" in a Windows username is not read as a pattern.
    # glob also matches folders; only files are candidates.
    matches = [
        path
        for path in glob.glob(os.path.join(glob.escape(directory), pattern))
        if os.path.isfile(path)
    ]
    if not matches:
        return None
    newest = max(matches, key=os.path.getmtime)
    if len(matches) > 1:
        print(
            f"  NOTE: {len(matches)} files match {pattern} in {directory}; "
            f"using the newest, {os.path.basename(newest)}."
        )
    return newest


def find_projections_file(
    override: Optional[str] = None, directory: Optional[str] = None
) -> str:
    """
    Picks the projections CSV this run optimizes on.

    Args:
        override: The filepath argument, used as given when supplied. It must
            name a file; a folder is rejected rather than searched.
        directory: Folder searched otherwise; defaults to DOWNLOADS_DIR.

    Returns:
        `override`, else the newest PROJECTIONS_GLOB match in the folder.

    Raises:
        FileNotFoundError: If `override` names no file, or no override was
            given and the folder holds no projections file.
    """
    if override:
        if os.path.isdir(override):
            raise FileNotFoundError(
                f"Expected a projections CSV file, not a folder: {override}"
            )
        if not os.path.isfile(override):
            raise FileNotFoundError(f"Projections file not found at: {override}")
        return override
    path = _newest_download(PROJECTIONS_GLOB, directory)
    if path is None:
        raise FileNotFoundError(
            f"No {PROJECTIONS_GLOB} file found in {directory or DOWNLOADS_DIR}. "
            f"Download your projections, or pass the file's path as the first "
            f"argument."
        )
    return path


def reject_stray_projections_path(
    filepath: Optional[str], *name_lists: Optional[List[str]]
) -> None:
    """
    Stops a run whose projections path was typed after -l / -x.

    Those flags take one or more values, so argparse files a path written
    after them under the flag and leaves `filepath` empty. Left alone, the run
    would skip the "player" and quietly optimize on the newest download.

    Raises:
        ValueError: If `filepath` is empty and a lock/exclude entry names a
            CSV or an existing file.
    """
    if filepath:
        return
    for names in name_lists:
        for name in names or []:
            if name.lower().endswith(".csv") or os.path.isfile(name):
                raise ValueError(
                    f"'{name}' was read as a player name for -l/-x, but it looks "
                    f"like a projections file. Put the projections path first, "
                    f"before any flags."
                )


def describe_modified(path: str) -> str:
    """Returns "modified Sat 09/19 11:20AM" for a file, so a stale download stands out."""
    return f"modified {datetime.fromtimestamp(os.path.getmtime(path)):%a %m/%d %I:%M%p}"


def find_dk_entries_file(
    override: Optional[str] = None, directory: Optional[str] = None
) -> Optional[str]:
    """
    Picks the DraftKings entries file this run reads.

    Args:
        override: The -dk / --dk-entries path, used as given when supplied.
        directory: Folder searched otherwise; defaults to DOWNLOADS_DIR.

    Returns:
        `override`, else the newest DK_ENTRIES_GLOB match in the folder, else
        None when the folder holds no entries file.

    Raises:
        FileNotFoundError: If `override` names no file. An explicit path is
            never silently swapped for another or skipped.
    """
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"DraftKings entries file not found: {override}")
        return override
    return _newest_download(DK_ENTRIES_GLOB, directory)


def _find_dk_pool(
    rows: List[List[str]],
) -> Optional[Tuple[Dict[str, int], List[List[str]]]]:
    """
    Locates the player pool section of a DraftKings entries file.

    The file is jagged: contest entries fill the leading columns and the pool
    sits to their right under its own "Name + ID" header, a few rows down. The
    whole file is scanned for that header, so a shifted instructions block
    cannot hide it. Column positions are read from the full row -- the
    entry-list columns to the pool's left never carry a pool header name -- so
    pool rows need no slicing.

    Returns:
        (header name -> column index, the rows below the header), or None when
        the file has no pool header. A repeated header name maps to its first
        column.
    """
    for idx, cells in enumerate(rows):
        if "Name + ID" in cells:
            columns: Dict[str, int] = {}
            for col, name in enumerate(cells):
                if name.strip():
                    columns.setdefault(name.strip(), col)
            return columns, rows[idx + 1 :]
    return None


def load_dk_name_ids(path: Optional[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Indexes every player's DraftKings "Name + ID" value from the entries file.

    Each player appears once per roster slot, so the Captain and FLEX versions
    of a Showdown player carry different IDs and must be looked up by slot.

    Args:
        path: Location of the DraftKings entries CSV (see find_dk_entries_file).

    Returns:
        A dict with "by_slot" (slot + name keys), "by_name" (name keys, with
        ambiguous names mapped to None), and "by_team" (slot + team keys, used
        for DST rows whose names differ between the two files). Returns None
        when there is no file or it carries no readable player pool, which
        tells the caller to omit the DraftKings upload rows entirely.
    """
    if not path or not os.path.exists(path):
        return None

    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
    except (OSError, csv.Error):
        return None

    pool = _find_dk_pool(rows)
    if pool is None:
        return None
    columns, pool_rows = pool
    if not {"Name + ID", "Name", "Roster Position"} <= columns.keys():
        return None
    name_id_col = columns["Name + ID"]
    name_col = columns["Name"]
    slot_col = columns["Roster Position"]
    team_col = columns.get("TeamAbbrev")
    position_col = columns.get("Position")

    by_slot: Dict[str, str] = {}
    by_name: Dict[str, Optional[str]] = {}
    by_team: Dict[str, str] = {}

    for cells in pool_rows:
        if len(cells) <= max(name_id_col, name_col, slot_col):
            continue
        name_id = cells[name_id_col].strip()
        name = _normalize_name(cells[name_col])
        slot = cells[slot_col].strip().upper()
        if not name_id or not name or not slot:
            continue

        for key in {name, _strip_name_suffix(name)}:
            by_slot.setdefault(f"{slot}|{key}", name_id)
            # A name that resolves to more than one player is unusable on its
            # own, so mark it ambiguous rather than guessing.
            if key in by_name and by_name[key] != name_id:
                by_name[key] = None
            else:
                by_name.setdefault(key, name_id)

        is_dst = position_col is not None and cells[position_col].strip().upper() in (
            "DST",
            "DEF",
            "D",
        )
        if is_dst and team_col is not None and len(cells) > team_col:
            team = _normalize_name(cells[team_col])
            if team:
                by_team.setdefault(f"{slot}|{team}", name_id)

    if not by_slot:
        return None
    return {"by_slot": by_slot, "by_name": by_name, "by_team": by_team}


def lookup_dk_name_id(
    lookup: Dict[str, Dict[str, Any]],
    player: Any,
    slot: Any,
    team: Any,
    position: Any,
) -> Optional[str]:
    """
    Resolves one rostered player to its slot-specific DraftKings "Name + ID".

    Tries the slot-qualified name first, then the suffix-stripped name, then an
    unambiguous name-only match, and finally the team abbreviation for defenses
    (DraftKings names them by nickname, projections rarely do).

    Returns:
        The "Name + ID" string, or None when no confident match exists.
    """
    name = _normalize_name(player)
    slot_key = str(slot).upper()
    candidates = [name, _strip_name_suffix(name)]

    for candidate in candidates:
        hit = lookup["by_slot"].get(f"{slot_key}|{candidate}")
        if hit:
            return hit
    for candidate in candidates:
        hit = lookup["by_name"].get(candidate)
        if hit:
            return hit
    if str(position).upper() in ("DST", "DEF", "D"):
        return lookup["by_team"].get(f"{slot_key}|{_normalize_name(team)}")
    return None


def build_dk_upload_values(
    rows: List[Dict[str, Any]], lookup: Optional[Dict[str, Dict[str, Any]]]
) -> Optional[List[str]]:
    """
    Converts a lineup's display rows into DraftKings "Name + ID" values.

    Returns:
        One value per roster slot in display order, or None when the entries
        file is unavailable or any player could not be matched -- in which case
        the caller omits the upload row for this lineup.
    """
    if not lookup:
        return None

    values: List[str] = []
    for row in rows:
        name_id = lookup_dk_name_id(
            lookup, row["Player"], row["Slot"], row["Team"], row["Position"]
        )
        if not name_id:
            print(
                f"  NOTE: No DraftKings ID found for {row['Player']} "
                f"({row['Slot']}); skipping the upload row for this lineup."
            )
            return None
        values.append(name_id)
    return values


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


def load_player_data(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses Showdown player data from the projections CSV file.

    Args:
        filepath: The absolute path to the projections CSV file.

    Returns:
        A pandas DataFrame containing cleaned and prepared player data ready
        for optimization, including derived Captain-slot columns.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty, critical columns are missing, or the
            slate does not consist of exactly two teams.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Projections file not found at: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The projections file is empty: {filepath}")

    print(f"Successfully loaded {len(df)} players from {os.path.basename(filepath)}.")

    # --- Data Cleaning and Preparation ---
    # Rename columns for consistency, applying only the aliases actually present
    # and never letting two of them collapse onto the same internal name.
    rename_map: Dict[str, str] = {}
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
        raise ValueError(
            f"Projections file is missing required column(s): {', '.join(missing)}"
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
        print("  NOTE: No 'Ceiling' column found. Ceiling values will display as 0.00.")
        df["Ceiling"] = 0.0
    if "Ownership" not in df.columns:
        print("  NOTE: No 'Total Own' column found. Ownership will display as 0.00%.")
        df["Ownership"] = 0.0
    if "CptOwnership" not in df.columns:
        print("  NOTE: No 'CPT Own' column found. Captain ownership treated as 0.00%.")
        df["CptOwnership"] = 0.0

    df["Ownership"] = df["Ownership"].fillna(0.0)
    df["CptOwnership"] = df["CptOwnership"].fillna(0.0)

    # Drop players with missing critical data for optimization.
    df.dropna(subset=["Player", "Position", "Team", "Salary", "Projection"], inplace=True)
    df["Salary"] = df["Salary"].astype(int)

    # Ceiling is filled only after that drop, so the count below describes the
    # players actually available to the optimizer. A blank ceiling is not a
    # reason to drop a player -- projection-only runs never touch the column --
    # but it is worth saying out loud, because it becomes a real zero and the
    # ceiling-weighted targets maximize exactly this number.
    missing_ceiling = int(df["Ceiling"].isna().sum())
    if missing_ceiling:
        print(
            f"  NOTE: {missing_ceiling} player(s) have no usable 'Ceiling' value; "
            f"treating it as 0.00."
        )
    df["Ceiling"] = df["Ceiling"].fillna(0.0)

    # --- Derived Captain-Slot Columns ---
    # Prefer the values supplied by the projections file; fall back to the
    # standard 1.5x Captain multiplier when a column is absent or blank.
    if "CptSalary" in df.columns:
        df["CptSalary"] = df["CptSalary"].fillna(df["Salary"] * CAPTAIN_MULTIPLIER)
    else:
        df["CptSalary"] = df["Salary"] * CAPTAIN_MULTIPLIER
    df["CptSalary"] = df["CptSalary"].round().astype(int)

    if "CptProjection" in df.columns:
        df["CptProjection"] = df["CptProjection"].fillna(
            df["Projection"] * CAPTAIN_MULTIPLIER
        )
    else:
        df["CptProjection"] = df["Projection"] * CAPTAIN_MULTIPLIER

    # Prefer a published "CPT Ceiling" and fall back to the 1.5x multiplier,
    # the same precedence CptSalary and CptProjection use. This matters now
    # that ceiling feeds the objective: a source whose Captain values are not
    # exactly 1.5x would otherwise have its Captain scaled one way under the
    # projection target and another under the ceiling target.
    if "CptCeiling" in df.columns:
        df["CptCeiling"] = df["CptCeiling"].fillna(df["Ceiling"] * CAPTAIN_MULTIPLIER)
    else:
        df["CptCeiling"] = df["Ceiling"] * CAPTAIN_MULTIPLIER

    # Slot-aware ownership: "Total Own" already includes "CPT Own", so a player's
    # FLEX-only ownership is the difference between the two.
    df["FlexOwnership"] = (df["Ownership"] - df["CptOwnership"]).clip(lower=0.0)

    # --- Slate Validation ---
    teams = sorted(df["Team"].astype(str).unique())
    if len(teams) != 2:
        raise ValueError(
            f"Showdown slates must contain exactly two teams, but {len(teams)} were "
            f"found: {', '.join(teams)}. Filter the projections file down to a "
            f"single game before optimizing."
        )

    print(
        f"Data preprocessed. {len(df)} players available for optimization "
        f"({teams[0]} vs {teams[1]})."
    )
    return df


def parse_player_selector(token: str) -> Tuple[str, Optional[str]]:
    """
    Splits a -l / -x argument into a player name and an optional slot qualifier.

    Accepts a bare name ("Drake Maye") meaning "any slot", or a name with a
    trailing ":CPT" / ":FLEX" suffix to target one specific roster slot.

    Args:
        token: The raw command-line token.

    Returns:
        A (player_name, slot) tuple where slot is "CPT", "FLEX", or None.
    """
    if ":" in token:
        name, _, suffix = token.rpartition(":")
        if name.strip() and suffix.strip().upper() in VALID_SLOTS:
            return name.strip(), suffix.strip().upper()
    return token.strip(), None


def _dedupe_selectors(tokens: Sequence[str]) -> List[Tuple[str, Optional[str]]]:
    """
    Parses -l / -x tokens into (name, slot) pairs, dropping exact repeats.

    Repeating a selector is a no-op for the model but would otherwise generate
    two PuLP constraints with the same name, which PuLP rejects outright.
    """
    seen = set()
    parsed: List[Tuple[str, Optional[str]]] = []
    for token in tokens:
        name, slot = parse_player_selector(token)
        key = (name.lower(), slot)
        if key in seen:
            print(f"  NOTE: Ignoring repeated selector '{token}'.")
            continue
        seen.add(key)
        parsed.append((name, slot))
    return parsed


def _find_player_indices(players_df: pd.DataFrame, player_name: str) -> pd.Index:
    """
    Returns the DataFrame indices matching a player name, case-insensitively.

    Names are matched without regard to team. Two players sharing a name on a
    single Showdown slate is vanishingly rare, but the selector would then apply
    to both of them, so say so rather than doing it silently.
    """
    matches = players_df[
        players_df["Player"].astype(str).str.strip().str.lower() == player_name.lower()
    ].index
    teams = sorted(players_df.loc[matches, "Team"].astype(str).unique())
    if len(teams) > 1:
        print(
            f"  WARNING: '{player_name}' matches players on more than one team "
            f"({', '.join(teams)}). The selector applies to all of them."
        )
    return matches


def build_lineup_rows(
    players_df: pd.DataFrame, captain_idx: Any, flex_indices: Sequence[Any]
) -> List[Dict[str, Any]]:
    """
    Converts a solved lineup into ordered, slot-adjusted display rows.

    The Captain is listed first with its 1.5x salary, projection, ceiling, and
    its Captain-specific ownership. The five FLEX players follow, ordered from
    highest to lowest FLEX salary.

    Args:
        players_df: The full player DataFrame.
        captain_idx: Index of the player selected at Captain.
        flex_indices: Indices of the five players selected at FLEX.

    Returns:
        A list of six dictionaries, one per roster slot, in display order.
    """
    captain = players_df.loc[captain_idx]
    rows: List[Dict[str, Any]] = [
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


def build_total_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Builds the summary row that follows a lineup's six roster rows in the export.

    Args:
        rows: The six display rows returned by build_lineup_rows().

    Returns:
        A row carrying the lineup's totals, with the player-identifying fields
        left blank so the summary reads clearly in a spreadsheet.
    """
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


def print_lineup(
    lineup_number: int,
    rows: List[Dict[str, Any]],
    target: str = TARGET_PROJECTION,
) -> None:
    """
    Prints a single Showdown lineup in a human-readable table.

    Args:
        lineup_number: 1-based lineup number used in the header.
        rows: The six display rows returned by build_lineup_rows().
        target: The active optimization target. A blended run's score matches
            neither printed total, so it gets its own line; the other targets
            are already shown by the Projection and Ceiling lines.
    """
    total_projection = sum(r["Projection"] for r in rows)
    total_ownership = sum(r["Ownership"] for r in rows)
    total_ceiling = sum(r["Ceiling"] for r in rows)
    total_salary = sum(r["Salary"] for r in rows)

    print(f"\n--- Optimal NFL Showdown Lineup #{lineup_number} ---")
    if target == TARGET_BLEND:
        # The weights come from OPTIMIZATION_TARGETS, the same place the
        # objective reads them, so a retuned blend can never print one number
        # while the solver maximizes another.
        label, proj_weight, ceiling_weight = OPTIMIZATION_TARGETS[target]
        print(
            f"Blend Score ({label}): "
            f"{proj_weight * total_projection + ceiling_weight * total_ceiling:.2f}"
        )
    print(f"Projection: {total_projection:.2f}")
    print(f"Total Ownership: {total_ownership:.2f}%")
    print(f"Ceiling: {total_ceiling:.2f}")
    print(f"Salary: ${total_salary:,} (${SALARY_CAP - total_salary:,} remaining)")
    print("-" * TABLE_WIDTH)
    print(
        f"{'Slot':<6} {'Player':<25} {'Pos':<5} {'Team':<6} "
        f"{'Salary':>8} {'Proj':>8} {'Own%':>8} {'Ceiling':>9}"
    )
    print("-" * TABLE_WIDTH)
    for row in rows:
        print(
            f"{row['Slot']:<6} {str(row['Player']):<25} {str(row['Position']):<5} "
            f"{str(row['Team']):<6} ${row['Salary']:>7,} "
            f"{row['Projection']:>8.2f} {row['Ownership']:>7.2f}% "
            f"{row['Ceiling']:>9.2f}"
        )
    print("-" * TABLE_WIDTH)


def main() -> None:
    """Main orchestrator function for the script."""
    parser = argparse.ArgumentParser(
        description="DraftKings NFL Showdown Multi-Lineup Optimizer."
    )
    parser.add_argument(
        "filepath",
        type=str,
        nargs="?",
        help=(
            f"Path to the DraftKings Showdown projections CSV file. Omit it to "
            f"use the newest {PROJECTIONS_GLOB} in Downloads."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-lineups",
        type=int,
        default=1,
        help="Number of unique lineups to generate (default: 1).",
    )
    parser.add_argument(
        "-u",
        "--min-uniques",
        type=int,
        default=1,
        help=(
            "Minimum number of unique roster spots between lineups (default: 1). "
            "Uniqueness is slot-aware, so the same six players with a different "
            "Captain counts as two uniques."
        ),
    )
    parser.add_argument(
        "-e",
        "--export",
        action="store_true",
        help="Export the generated lineups to a CSV file.",
    )
    parser.add_argument(
        "-l",
        "--lock",
        nargs="+",
        help=(
            "Player names to lock into the lineup (case-insensitive). Append "
            "':CPT' or ':FLEX' to lock a player into a specific slot, "
            'e.g. -l "Drake Maye:CPT" "A.J. Brown".'
        ),
    )
    parser.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        help=(
            "Player names to exclude from the lineup (case-insensitive). Append "
            "':CPT' or ':FLEX' to ban a player from only that slot, "
            'e.g. -x "Sam Darnold:CPT".'
        ),
    )
    parser.add_argument(
        "-ms",
        "--max-salary",
        type=int,
        default=SALARY_CAP,
        help=(
            f"Maximum total lineup salary (default: {SALARY_CAP:,}). Values above "
            f"the ${SALARY_CAP:,} DraftKings cap are clamped to the cap."
        ),
    )
    parser.add_argument(
        "-mns",
        "--min-salary",
        type=int,
        default=0,
        help=(
            "Minimum total lineup salary (default: no floor). It must not "
            "exceed --max-salary."
        ),
    )
    # The scoring target the solver maximizes. Passing neither flag keeps the
    # long-standing projection-only behavior.
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "-c",
        "--ceiling",
        dest="ceiling",
        action="store_true",
        help="Optimize on ceiling instead of projection.",
    )
    target_group.add_argument(
        "-pj",
        "--projceiling",
        dest="projceiling",
        action="store_true",
        help="Optimize on an equally weighted 50/50 blend of projection and ceiling.",
    )
    parser.add_argument(
        "-dk",
        "--dk-entries",
        metavar="PATH",
        help=(
            f"DraftKings entries CSV for the export's upload rows, instead of "
            f"the newest {DK_ENTRIES_GLOB} in Downloads."
        ),
    )
    args = parser.parse_args()

    try:
        # --- 1. Validate arguments ---
        if args.num_lineups < 1:
            raise ValueError("--num-lineups must be at least 1.")
        if not 1 <= args.min_uniques <= ROSTER_SIZE:
            raise ValueError(
                f"--min-uniques must be between 1 and {ROSTER_SIZE} (roster size)."
            )
        max_salary = min(args.max_salary, SALARY_CAP)
        if args.max_salary > SALARY_CAP:
            print(
                f"\nNOTE: --max-salary ${args.max_salary:,} exceeds the DraftKings "
                f"cap. Clamping to ${SALARY_CAP:,}."
            )
        if max_salary <= 0:
            raise ValueError("--max-salary must be a positive number.")
        if args.min_salary < 0:
            raise ValueError("--min-salary cannot be negative.")
        if args.min_salary > max_salary:
            raise ValueError(
                f"--min-salary ${args.min_salary:,} cannot exceed the lineup's "
                f"maximum salary of ${max_salary:,}."
            )

        target = resolve_optimization_target(args.ceiling, args.projceiling)
        target_label = OPTIMIZATION_TARGETS[target][0]

        # --- 2. Load and prepare data ---
        reject_stray_projections_path(args.filepath, args.lock, args.exclude)
        projections_path = find_projections_file(args.filepath)
        print(f"Projections file: {projections_path} ({describe_modified(projections_path)})")
        players_df = load_player_data(projections_path)
        print(f"Optimizing on: {target_label}.")
        validate_target_data(players_df, target)
        players_dict = players_df.to_dict("index")
        player_indices = list(players_dict.keys())
        teams = sorted(players_df["Team"].astype(str).unique())

        # --- 3. Define the Optimization Problem (once) ---
        prob = pulp.LpProblem("DraftKings_NFL_Showdown_Multi_Lineup", pulp.LpMaximize)
        # Two independent binary decisions per player: Captain and FLEX.
        cpt_vars = pulp.LpVariable.dicts("CPT", player_indices, cat="Binary")
        flex_vars = pulp.LpVariable.dicts("FLEX", player_indices, cat="Binary")

        # --- 4. Objective and Base Constraints (once) ---
        # The objective is a weighted mix of projection and ceiling, evaluated
        # per slot so the Captain contributes its 1.5x values under any target.
        prob += (
            pulp.lpSum(
                target_value(players_dict[i], target, captain=True) * cpt_vars[i]
                + target_value(players_dict[i], target, captain=False) * flex_vars[i]
                for i in player_indices
            ),
            "Total_Target_Value",
        )

        # Salary cap (respects --max-salary, which never exceeds the DK cap)
        # and the --min-salary floor, bounding the same expression from both
        # sides so the Captain's 1.5x salary counts once either way. The
        # default floor of 0 adds no constraint at all.
        salary_expr = pulp.lpSum(
            players_dict[i]["CptSalary"] * cpt_vars[i]
            + players_dict[i]["Salary"] * flex_vars[i]
            for i in player_indices
        )
        prob += (salary_expr <= max_salary, "Salary_Cap")
        if args.min_salary > 0:
            print(f"\nEnforcing a minimum lineup salary of ${args.min_salary:,}...")
            prob += (salary_expr >= args.min_salary, "Min_Salary")
        # Exactly one Captain
        prob += (
            pulp.lpSum(cpt_vars[i] for i in player_indices) == 1,
            "Captain_Slot",
        )
        # Exactly five FLEX
        prob += (
            pulp.lpSum(flex_vars[i] for i in player_indices) == FLEX_SLOTS,
            "Flex_Slots",
        )
        # A player cannot be rostered at both Captain and FLEX
        for i in player_indices:
            prob += (cpt_vars[i] + flex_vars[i] <= 1, f"One_Slot_Per_Player_{i}")
        # A player listed on more than one row (duplicate projections export)
        # must still occupy at most one roster spot.
        duplicate_groups = {
            key: idxs
            for key, idxs in players_df.groupby(
                [
                    players_df["Player"].astype(str).str.strip().str.lower(),
                    players_df["Team"].astype(str),
                ]
            ).groups.items()
            if len(idxs) > 1
        }
        if duplicate_groups:
            print(
                f"\nNOTE: {len(duplicate_groups)} player(s) appear on multiple rows. "
                f"Constraining each to at most one roster spot:"
            )
            for (name, team), idxs in duplicate_groups.items():
                print(f"  {name} ({team}) - {len(idxs)} rows")
                prob += (
                    pulp.lpSum(cpt_vars[i] + flex_vars[i] for i in idxs) <= 1,
                    f"One_Row_Per_Player_{_safe_name(name)}_{_safe_name(team)}",
                )

        # Lineups must include at least one player from each team
        for team in teams:
            team_indices = players_df[players_df["Team"].astype(str) == team].index
            prob += (
                pulp.lpSum(cpt_vars[i] + flex_vars[i] for i in team_indices) >= 1,
                f"Min_One_From_{_safe_name(team)}",
            )

        # --- Locking Players ---
        if args.lock:
            print(f"\nLocking players: {args.lock}")
            for player_name, slot in _dedupe_selectors(args.lock):
                matches = _find_player_indices(players_df, player_name)
                if matches.empty:
                    print(
                        f"  WARNING: Player '{player_name}' not found in projections. "
                        f"Skipping lock."
                    )
                    continue

                tag = _safe_name(f"{player_name}_{slot or 'ANY'}")
                if slot == SLOT_CPT:
                    expression = pulp.lpSum(cpt_vars[i] for i in matches)
                elif slot == SLOT_FLEX:
                    expression = pulp.lpSum(flex_vars[i] for i in matches)
                else:
                    expression = pulp.lpSum(
                        cpt_vars[i] + flex_vars[i] for i in matches
                    )
                prob += (expression == 1, f"Lock_{tag}")
                print(
                    f"  Locked: {players_df.loc[matches[0], 'Player']} "
                    f"@ {slot or 'ANY SLOT'}"
                )

        # --- Excluding Players ---
        if args.exclude:
            print(f"\nExcluding players: {args.exclude}")
            for player_name, slot in _dedupe_selectors(args.exclude):
                matches = _find_player_indices(players_df, player_name)
                if matches.empty:
                    print(
                        f"  WARNING: Player '{player_name}' not found in projections. "
                        f"Skipping exclusion."
                    )
                    continue

                for idx in matches:
                    tag = _safe_name(f"{idx}_{slot or 'ANY'}")
                    if slot == SLOT_CPT:
                        prob += (cpt_vars[idx] == 0, f"Exclude_{tag}")
                    elif slot == SLOT_FLEX:
                        prob += (flex_vars[idx] == 0, f"Exclude_{tag}")
                    else:
                        prob += (
                            cpt_vars[idx] + flex_vars[idx] == 0,
                            f"Exclude_{tag}",
                        )
                    print(
                        f"  Excluded: {players_df.loc[idx, 'Player']} "
                        f"@ {slot or 'ANY SLOT'}"
                    )

        # --- 5. Iterative Optimization Loop ---
        solver = pulp.HiGHS(msg=False)
        if not solver.available():
            raise ValueError(
                "The HiGHS solver is not available. Install it with: pip install highspy"
            )

        max_slots_can_share = ROSTER_SIZE - args.min_uniques
        all_lineups_export_data: List[Dict[str, Any]] = []

        # DraftKings "Name + ID" values for the upload row that follows each
        # lineup's totals. Absent or unreadable entries file: no upload rows.
        entries_path = find_dk_entries_file(args.dk_entries) if args.export else None
        dk_lookup = load_dk_name_ids(entries_path) if args.export else None
        if args.export and dk_lookup is None:
            print(
                f"\nNOTE: No readable DraftKings entries file "
                f"({entries_path or f'no {DK_ENTRIES_GLOB} in {DOWNLOADS_DIR}'}). "
                f"The export will omit the upload rows."
            )

        for i in range(args.num_lineups):
            print(f"\n--- Generating Lineup #{i + 1} ---")

            prob.solve(solver)
            status = pulp.LpStatus[prob.status]

            if status != "Optimal":
                print(f"Could not find an optimal lineup. Status: {status}")
                if i == 0:
                    print(
                        "This means no lineup exists that satisfies the constraints."
                    )
                    if (
                        args.lock
                        or args.exclude
                        or max_salary < SALARY_CAP
                        or args.min_salary > 0
                    ):
                        print(
                            "  Check your --lock / --exclude selections and "
                            "--max-salary / --min-salary; they are the usual cause."
                        )
                else:
                    print(f"Stopped after generating {i} unique lineups.")
                    if args.min_salary > 0:
                        print(
                            f"  The ${args.min_salary:,} --min-salary floor shrinks "
                            f"the pool; lowering it yields more lineups."
                        )
                break

            captain_idx = next(
                idx
                for idx in player_indices
                if (cpt_vars[idx].varValue or 0) > 0.5
            )
            flex_indices = [
                idx
                for idx in player_indices
                if (flex_vars[idx].varValue or 0) > 0.5
            ]

            # Diversity constraint: this exact set of roster spots (slot-aware)
            # may not repeat in any future lineup. Promoting a FLEX to Captain
            # therefore counts as two unique roster spots.
            prob += (
                cpt_vars[captain_idx]
                + pulp.lpSum(flex_vars[idx] for idx in flex_indices)
                <= max_slots_can_share,
                f"Diversity_from_lineup_{i + 1}",
            )

            # --- 6. Display the current lineup ---
            rows = build_lineup_rows(players_df, captain_idx, flex_indices)
            print_lineup(i + 1, rows, target)

            # Collect data for export if requested
            if args.export:
                for row in rows:
                    export_row = row.copy()
                    export_row["Lineup_ID"] = i + 1
                    all_lineups_export_data.append(export_row)
                total_row = build_total_row(rows)
                total_row["Lineup_ID"] = i + 1
                all_lineups_export_data.append(total_row)

                # DraftKings upload row: the same lineup laid out horizontally,
                # one "Name + ID" per roster slot in CPT-then-FLEX order.
                dk_values = build_dk_upload_values(rows, dk_lookup)
                if dk_values:
                    dk_row: Dict[str, Any] = {"Lineup_ID": i + 1}
                    dk_row.update(zip(EXPORT_COLUMNS[1:], dk_values))
                    all_lineups_export_data.append(dk_row)

        # --- 7. Export All Lineups to CSV ---
        if args.export and all_lineups_export_data:
            os.makedirs(EXPORT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = (
                f"nfl_showdown_multi_lineups"
                f"{TARGET_FILE_SUFFIXES[target]}_{timestamp}.csv"
            )
            export_path = os.path.join(EXPORT_DIR, filename)

            export_df = pd.DataFrame(all_lineups_export_data)
            export_df = export_df.reindex(columns=EXPORT_COLUMNS)
            # Round derived floats so the export doesn't carry binary-float
            # noise. The upload rows put strings in these columns, so round
            # per value rather than over the whole column.
            for col in ["Projection", "Ownership", "Ceiling"]:
                export_df[col] = export_df[col].map(
                    lambda v: round(v, 2) if isinstance(v, float) else v
                )

            export_df.to_csv(export_path, index=False)
            print(f"\nAll generated lineups exported to: {export_path}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nFATAL ERROR: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {type(e).__name__}: {e}")
        # Send the traceback to stderr so stdout stays clean for the lineups
        # while a bug report still carries something actionable.
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Helpers shared by the Classic and Showdown optimizers.

Nothing here prints. Functions that have something to tell the user take an
optional `notes` list and append the message to it; the caller decides where it
goes (the CLI prints it, the app shows it).

EXPORT_DIR and DOWNLOADS_DIR are read at call time, never bound as defaults,
so tests and tools can point them elsewhere by assigning the module attribute.
"""

import csv
import glob
import os
import re
from datetime import datetime
from typing import Any

import pandas as pd
import pulp

SALARY_CAP: int = 50000

EXPORT_DIR: str = r"G:\My Drive\Documents\NFL-DFS\csv-exports"
# John Rankin's (Dad) Downloads folder path: r"C:\Users\jdr0824\Downloads"

# DraftKings entries file: it supplies the "Name + ID" values of the export's
# upload row, the kickoffs that seat the Classic FLEX, and the entries late
# swap rebuilds. -dk / --dk-entries names one explicitly; otherwise the newest
# match in the current user's Downloads folder wins, so a browser re-download
# ("DKEntries (1).csv") is picked up. Files that merely contain the name
# ("SD-DKEntries.csv", "upload-ready-DKEntries-...") do not match.
DOWNLOADS_DIR: str = os.path.join(os.path.expanduser("~"), "Downloads")
DK_ENTRIES_GLOB: str = "DKEntries*.csv"

# Name suffixes dropped when a projections name and a DraftKings name disagree.
NAME_SUFFIXES: tuple[str, ...] = ("jr", "sr", "ii", "iii", "iv", "v")

# --- Optimization Targets ---
# Each target weights a player's projection and ceiling into the single value
# the solver maximizes. Projection-only is the default and reproduces the
# behavior the optimizers had before the other targets existed.
TARGET_PROJECTION: str = "projection"
TARGET_CEILING: str = "ceiling"
TARGET_BLEND: str = "blend"

# target -> (label for output, projection weight, ceiling weight)
OPTIMIZATION_TARGETS: dict[str, tuple[str, float, float]] = {
    TARGET_PROJECTION: ("Projection", 1.0, 0.0),
    TARGET_CEILING: ("Ceiling", 0.0, 1.0),
    TARGET_BLEND: ("50/50 Projection + Ceiling", 0.5, 0.5),
}

# Suffix added to the export filename so a run's target is obvious on disk.
TARGET_FILE_SUFFIXES: dict[str, str] = {
    TARGET_PROJECTION: "",
    TARGET_CEILING: "_ceiling",
    TARGET_BLEND: "_projceiling",
}

EXPORT_TIMESTAMP_FORMAT: str = "%Y-%m-%d_%H-%M-%S"


def resolve_optimization_target(use_ceiling: bool, use_blend: bool) -> str:
    """
    Turns the --ceiling / --projceiling flags into a single target key.

    Raises:
        ValueError: If both flags are supplied together. argparse's mutually
            exclusive group rejects that first, so a CLI user never reaches
            this; it keeps the helper correct when called from elsewhere.
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


def target_score(target: str, projection: float, ceiling: float) -> float:
    """The objective value a lineup with these totals scores under `target`."""
    _, proj_weight, ceiling_weight = OPTIMIZATION_TARGETS[target]
    return proj_weight * projection + ceiling_weight * ceiling


def check_target_data(df: pd.DataFrame, target: str) -> str | None:
    """
    Checks the ceiling data a ceiling-weighted target is about to maximize.

    An all-zero Ceiling column is fatal: optimizing on it would silently return
    an arbitrary salary-feasible lineup. A partly-populated column is legal but
    consequential -- a zero ceiling is indistinguishable from a blank one once
    filled, and those players score nothing on the ceiling half of the
    objective -- so name them rather than letting the pool shrink invisibly.

    Returns:
        A warning naming the zero-ceiling players, or None.

    Raises:
        ValueError: If the target weights ceiling but no positive ceiling exists.
    """
    label, _, ceiling_weight = OPTIMIZATION_TARGETS[target]
    if ceiling_weight == 0:
        return None
    if "Ceiling" not in df.columns or not (df["Ceiling"] > 0).any():
        raise ValueError(
            f"The '{label}' target needs a populated 'Ceiling' column, but the "
            f"projections file has no ceiling values. Re-run without "
            f"--ceiling/--projceiling to optimize on projection."
        )

    zeroed = df[df["Ceiling"] <= 0]
    if zeroed.empty:
        return None
    consequence = (
        "they cannot be rostered unless locked"
        if target == TARGET_CEILING
        else "they are scored on their projection alone"
    )
    names = ", ".join(str(name) for name in zeroed["Player"].head(5))
    if len(zeroed) > 5:
        names += f", +{len(zeroed) - 5} more"
    return (
        f"  WARNING: {len(zeroed)} of {len(df)} players have a zero or missing "
        f"ceiling. Under the '{label}' target {consequence}: {names}"
    )


def build_solver() -> pulp.LpSolver:
    """
    Returns the HiGHS solver every model is solved with.

    One instance serves a whole run: PuLP rebuilds the underlying HiGHS model
    from the problem on each solve, so the same solver object is safe to reuse
    both across the multi-lineup loop's accumulated cuts and across the
    one-problem-per-entry models late swap builds.

    Raises:
        ValueError: If highspy is not installed, which is the only way HiGHS
            can be unavailable.
    """
    solver = pulp.HiGHS(msg=False)
    if not solver.available():
        raise ValueError(
            "The HiGHS solver is not available. Install it with: pip install highspy"
        )
    return solver


# --- Files ---


def _newest_download(
    pattern: str, directory: str | None = None, notes: list[str] | None = None
) -> str | None:
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
    if len(matches) > 1 and notes is not None:
        notes.append(
            f"  NOTE: {len(matches)} files match {pattern} in {directory}; "
            f"using the newest, {os.path.basename(newest)}."
        )
    return newest


def find_projections_file(
    pattern: str,
    override: str | None = None,
    directory: str | None = None,
    notes: list[str] | None = None,
) -> str:
    """
    Picks the projections CSV a run optimizes on.

    Args:
        pattern: The glob the format's projections downloads match.
        override: A path given explicitly, used as given. It must name a file;
            a folder is rejected rather than searched.
        directory: Folder searched otherwise; defaults to DOWNLOADS_DIR.

    Returns:
        `override`, else the newest `pattern` match in the folder.

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
    path = _newest_download(pattern, directory, notes)
    if path is None:
        raise FileNotFoundError(
            f"No {pattern} file found in {directory or DOWNLOADS_DIR}. "
            f"Download your projections, or pass the file's path as the first "
            f"argument."
        )
    return path


def find_dk_entries_file(
    override: str | None = None,
    directory: str | None = None,
    notes: list[str] | None = None,
) -> str | None:
    """
    Picks the DraftKings entries file a run reads.

    Returns:
        `override`, else the newest DK_ENTRIES_GLOB match in the folder
        (default DOWNLOADS_DIR), else None when the folder holds none.

    Raises:
        FileNotFoundError: If `override` names no file. An explicit path is
            never silently swapped for another or skipped.
    """
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"DraftKings entries file not found: {override}")
        return override
    return _newest_download(DK_ENTRIES_GLOB, directory, notes)


def describe_modified(path: str) -> str:
    """Returns "modified Sat 09/19 11:20AM" for a file, so a stale download stands out."""
    modified = datetime.fromtimestamp(os.path.getmtime(path))  # noqa: DTZ006 -- local time is the point
    return f"modified {modified:%a %m/%d %I:%M%p}"


def export_path(stem: str, target: str, now: datetime | None = None) -> str:
    """
    The file an export writes: EXPORT_DIR/<stem><target suffix>_<timestamp>.csv.
    """
    timestamp = (now or datetime.now()).strftime(EXPORT_TIMESTAMP_FORMAT)  # noqa: DTZ005 -- local
    filename = f"{stem}{TARGET_FILE_SUFFIXES[target]}_{timestamp}.csv"
    return os.path.join(EXPORT_DIR, filename)


# --- DraftKings entries file: the player pool and "Name + ID" lookups ---


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


def _find_dk_pool(
    rows: list[list[str]],
) -> tuple[dict[str, int], list[list[str]]] | None:
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
            columns: dict[str, int] = {}
            for col, name in enumerate(cells):
                if name.strip():
                    columns.setdefault(name.strip(), col)
            return columns, rows[idx + 1 :]
    return None


def load_dk_name_ids(path: str | None) -> dict[str, dict[str, Any]] | None:
    """
    Indexes every player's DraftKings "Name + ID" value from the entries file.

    Each player appears once per roster slot, so the Captain and FLEX versions
    of a Showdown player carry different IDs and must be looked up by slot.

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

    by_slot: dict[str, str] = {}
    by_name: dict[str, str | None] = {}
    by_team: dict[str, str] = {}

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


def _dk_roster_position(slot: Any) -> str:
    """
    Maps a display slot to the roster position DraftKings uses.

    The Classic display slots number the repeated positions (RB1, RB2, WR3),
    while the entries file lists them unnumbered. Showdown's CPT and FLEX pass
    through unchanged.
    """
    return re.sub(r"\d+$", "", str(slot)).upper()


def lookup_dk_name_id(
    lookup: dict[str, dict[str, Any]],
    player: Any,
    slot: Any,
    team: Any,
    position: Any,
) -> str | None:
    """
    Resolves one rostered player to its slot-specific DraftKings "Name + ID".

    Tries the slot-qualified name first, then the suffix-stripped name, then an
    unambiguous name-only match, and finally the team abbreviation for defenses
    (DraftKings names them by nickname, projections rarely do).

    Returns:
        The "Name + ID" string, or None when no confident match exists.
    """
    name = _normalize_name(player)
    slot_key = _dk_roster_position(slot)
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
    rows: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]] | None,
    notes: list[str] | None = None,
) -> list[str] | None:
    """
    Converts a lineup's rows (each with Player, Slot, Team, Position) into
    DraftKings "Name + ID" values, one per roster slot in the rows' order.

    Returns:
        The values, or None when the entries file is unavailable or any player
        could not be matched -- in which case the caller omits the upload row
        for this lineup (and a note says which player failed).
    """
    if not lookup:
        return None

    values: list[str] = []
    for row in rows:
        name_id = lookup_dk_name_id(
            lookup, row["Player"], row["Slot"], row["Team"], row["Position"]
        )
        if not name_id:
            if notes is not None:
                notes.append(
                    f"  NOTE: No DraftKings ID found for {row['Player']} "
                    f"({row['Slot']}); skipping the upload row for this lineup."
                )
            return None
        values.append(name_id)
    return values


def missing_upload_rows_note(entries_path: str | None) -> str:
    """The note an export prints when it has no readable entries file."""
    return (
        f"\nNOTE: No readable DraftKings entries file "
        f"({entries_path or f'no {DK_ENTRIES_GLOB} in {DOWNLOADS_DIR}'}). "
        f"The export will omit the upload rows."
    )


class PlayerDataError(ValueError):
    """
    A projections file that cannot be optimized on.

    `notes` carries whatever the loader had to report before it gave up, so the
    CLI can print it ahead of the error exactly as it always has. The Classic
    loader's notes depend on the ownership field, so there it is a dict keyed
    by field, as on classic.PlayerPool.
    """

    def __init__(
        self, message: str, notes: list[str] | dict[str, list[str]] | None = None
    ) -> None:
        super().__init__(message)
        self.notes = notes or []

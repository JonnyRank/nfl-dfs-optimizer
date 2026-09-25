"""
Logic behind the Streamlit app (app.py), kept free of Streamlit so it can be
tested directly: file discovery, the player grid, lock/exclude bookkeeping,
building options, and running + exporting a request.

Locks and excludes are held as {player key: slot}. The key survives a
re-downloaded projections file: the DraftKings ID for Classic, and
"name|team" for Showdown (whose files carry no ID). The slot is SLOT_ANY for
every Classic selection; Showdown also allows "CPT" and "FLEX".

Projection edits are held the same way, as {player key: {grid column: value}},
and exist only in memory: apply_overrides() lays them over a copy of the
loaded frame for the grid and for each run, and the source file is never
written. A Showdown FLEX Projection or Ceiling edit resets that player's
Captain value to 1.5x the new number, replacing any CPT value from the file.
"""

import glob
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from nfl_dfs_optimizer import classic, common, showdown
from nfl_dfs_optimizer.common import (
    SALARY_CAP,
    TARGET_BLEND,
    TARGET_CEILING,
    TARGET_PROJECTION,
    load_dk_name_ids,
    missing_upload_rows_note,
    target_score,
)

CLASSIC = "Classic"
SHOWDOWN = "Showdown"
FORMATS = (CLASSIC, SHOWDOWN)

PROJECTIONS_GLOBS = {CLASSIC: classic.PROJECTIONS_GLOB, SHOWDOWN: showdown.PROJECTIONS_GLOB}

# Radio label -> target key.
TARGET_CHOICES: dict[str, str] = {
    "Projection": TARGET_PROJECTION,
    "Ceiling": TARGET_CEILING,
    "50/50": TARGET_BLEND,
}
OWNERSHIP_CHOICES: dict[str, str] = {
    "Large field": classic.OWNERSHIP_LARGE_FIELD,
    "Small field": classic.OWNERSHIP_SMALL_FIELD,
}

SLOT_ANY = "Any"
SHOWDOWN_SLOTS = (SLOT_ANY, showdown.SLOT_CPT, showdown.SLOT_FLEX)
LOCK = "Lock"
EXCLUDE = "Exclude"
EDITED = "Edited"

Selections = dict[Any, str]
Edits = dict[Any, dict[str, float]]

# Editable grid column -> the loaded frame's column it overrides.
EDITABLE_COLUMNS: dict[str, dict[str, str]] = {
    CLASSIC: {
        "Projection": "Projection",
        "Ceiling": "Ceiling",
        "Small Field Own": classic.OWNERSHIP_COLUMNS[classic.OWNERSHIP_SMALL_FIELD],
        "Large Field Own": classic.OWNERSHIP_COLUMNS[classic.OWNERSHIP_LARGE_FIELD],
    },
    SHOWDOWN: {
        "Projection": "Projection",
        "Ceiling": "Ceiling",
        "Own": "Ownership",
        "CPT Own": "CptOwnership",
    },
}
# Showdown FLEX column -> the Captain column an edit to it recalculates at 1.5x.
CAPTAIN_FROM_FLEX: dict[str, str] = {"Projection": "CptProjection", "Ceiling": "CptCeiling"}
CAPTAIN_GRID_COLUMNS: dict[str, str] = {"Projection": "CPT Projection", "Ceiling": "CPT Ceiling"}
OWNERSHIP_GRID_COLUMNS = ("Small Field Own", "Large Field Own", "Own", "CPT Own")
# The grid shows two decimals, so a value within half a cent of another is
# the same value to the user (typing the file's number back undoes an edit).
EDIT_TOLERANCE = 0.005
EDITED_STYLE = "background-color: rgba(255, 196, 0, 0.28)"


# --- Files ---


def downloads_matching(pattern: str) -> list[str]:
    """Files in Downloads matching `pattern`, newest first."""
    directory = common.DOWNLOADS_DIR
    matches = [
        path
        for path in glob.glob(os.path.join(glob.escape(directory), pattern))
        if os.path.isfile(path)
    ]
    return sorted(matches, key=os.path.getmtime, reverse=True)


def projection_files(fmt: str) -> list[str]:
    return downloads_matching(PROJECTIONS_GLOBS[fmt])


def entries_files() -> list[str]:
    return downloads_matching(common.DK_ENTRIES_GLOB)


def describe_file(path: str) -> str:
    """ "name.csv (modified Sat 09/19 11:20AM)" """
    return f"{os.path.basename(path)} ({common.describe_modified(path)})"


def browse_for_csv(title: str, start_in: str | None = None) -> str | None:
    """
    Opens the Windows "Open" dialog and returns the chosen file's path, or
    None when cancelled. A browser never reveals a picked file's real path,
    but this app's server runs on the same PC, so it opens the dialog itself.
    The dialog is kept on top so it can't hide behind the browser. Blocks
    until the dialog closes.

    Raises:
        RuntimeError: If no dialog can be shown (no display, no Tk).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("The file dialog needs tkinter, which is not installed.") from exc
    start = start_in if start_in and os.path.isdir(start_in) else common.DOWNLOADS_DIR
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(f"Couldn't open the file dialog: {exc}") from exc
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            parent=root,
            title=title,
            initialdir=start,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    except tk.TclError as exc:
        raise RuntimeError(f"Couldn't open the file dialog: {exc}") from exc
    finally:
        root.destroy()
    return os.path.normpath(path) if path else None


def load_pool(fmt: str, path: str) -> Any:
    """The format's PlayerPool for a projections file (raises on a bad file)."""
    return (classic if fmt == CLASSIC else showdown).load_player_data(path)


def load_notes(fmt: str, pool: Any, ownership_field: str) -> list[str]:
    notes = pool.notes[ownership_field] if fmt == CLASSIC else pool.notes
    return [note.strip() for note in notes]


# --- Players and the grid ---


def player_key(fmt: str, row: pd.Series) -> Any:
    """The key a lock/exclude is stored under; stable across re-downloads."""
    if fmt == CLASSIC:
        return int(row["ID"])
    return f"{str(row['Player']).strip().lower()}|{row['Team']}"


def player_keys(fmt: str, df: pd.DataFrame) -> pd.Series:
    """player_key() for every row, indexed like `df`."""
    if fmt == CLASSIC:
        return df["ID"].astype(int)
    return df["Player"].astype(str).str.strip().str.lower() + "|" + df["Team"].astype(str)


def showdown_opponents(df: pd.DataFrame) -> pd.Series:
    """Showdown files have no Opp column; with exactly two teams it is the other one."""
    teams = sorted(df["Team"].astype(str).unique())
    other = {teams[0]: teams[-1], teams[-1]: teams[0]}
    return df["Team"].astype(str).map(other)


def apply_overrides(fmt: str, df: pd.DataFrame, edits: Edits | None) -> pd.DataFrame:
    """
    A copy of the loaded frame with the in-memory edits laid over it. The
    frame passed in -- and the file it came from -- are never modified.

    Showdown: an edited Projection or Ceiling sets the Captain value to 1.5x
    the new number (replacing a file-supplied CPT value), and FLEX ownership
    is recomputed from the possibly edited Own / CPT Own.
    """
    out = df.copy()
    if not edits:
        return out
    columns = EDITABLE_COLUMNS[fmt]
    for row, key in player_keys(fmt, df).items():
        for grid_column, value in edits.get(key, {}).items():
            out.at[row, columns[grid_column]] = value
            if fmt == SHOWDOWN and grid_column in CAPTAIN_FROM_FLEX:
                out.at[row, CAPTAIN_FROM_FLEX[grid_column]] = value * showdown.CAPTAIN_MULTIPLIER
    if fmt == CLASSIC:
        out["Ownership"] = out[classic.OWNERSHIP_COLUMNS[classic.OWNERSHIP_LARGE_FIELD]]
    else:
        out["FlexOwnership"] = showdown.flex_ownership(out)
    return out


def grid_frame(
    fmt: str,
    df: pd.DataFrame,
    locks: Selections,
    excludes: Selections,
    edits: Edits | None = None,
) -> pd.DataFrame:
    """
    The player grid, indexed like `df`: Lock and Exclude first (checkboxes for
    Classic, blank/Any/CPT/FLEX for Showdown), then the player columns with
    `edits` applied, then "Edited" naming each row's edited columns. `df` is
    the frame as loaded; the edits are laid over it here.
    """
    edits = edits or {}
    df = apply_overrides(fmt, df, edits)
    keys = player_keys(fmt, df)
    frame = pd.DataFrame(index=df.index)
    if fmt == CLASSIC:
        frame[LOCK] = keys.map(lambda k: k in locks)
        frame[EXCLUDE] = keys.map(lambda k: k in excludes)
        opp = df["Opp"].astype(str)
    else:
        frame[LOCK] = keys.map(lambda k: locks.get(k)).astype(object)
        frame[EXCLUDE] = keys.map(lambda k: excludes.get(k)).astype(object)
        opp = showdown_opponents(df)
    frame["Player"] = df["Player"].astype(str).str.strip()
    frame["Position"] = df["Position"]
    frame["Team"] = df["Team"]
    frame["Opp"] = opp
    frame["Salary"] = df["Salary"].astype(int)
    frame["Projection"] = df["Projection"]
    frame["Ceiling"] = df["Ceiling"]
    if fmt == CLASSIC:
        frame["Small Field Own"] = df[classic.OWNERSHIP_COLUMNS[classic.OWNERSHIP_SMALL_FIELD]]
        frame["Large Field Own"] = df[classic.OWNERSHIP_COLUMNS[classic.OWNERSHIP_LARGE_FIELD]]
    else:
        frame["CPT Salary"] = df["CptSalary"].astype(int)
        frame["CPT Projection"] = df["CptProjection"]
        frame["CPT Ceiling"] = df["CptCeiling"]
        frame["Own"] = df["Ownership"]
        frame["CPT Own"] = df["CptOwnership"]
    frame[EDITED] = keys.map(lambda k: ", ".join(edits.get(k, {})))
    return frame


def grid_styles(fmt: str, frame: pd.DataFrame) -> pd.DataFrame:
    """
    CSS for the grid: an edited row's Player cell, and for Showdown the Captain
    cells its edit recalculated. Streamlit styles only non-editable columns,
    so the edited cells themselves are named in the Edited column instead.
    """
    styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
    for row, edited in frame[EDITED].items():
        if not edited:
            continue
        styles.at[row, "Player"] = EDITED_STYLE
        styles.at[row, EDITED] = EDITED_STYLE
        if fmt == SHOWDOWN:
            for flex_column, captain_column in CAPTAIN_GRID_COLUMNS.items():
                if flex_column in edited.split(", "):
                    styles.at[row, captain_column] = EDITED_STYLE
    return styles


def grid_key(
    version: int,
    fmt: str,
    path: str,
    modified: float,
    positions: list[str],
    teams: list[str],
    search: str,
) -> str:
    """
    The data_editor key for one grid. Its pending edits are row positions, so
    anything that can change which player sits at a position -- the file, a
    re-save of it (mtime), the filters -- must change the key, or a leftover
    edit would land on whoever moved into that row.
    """
    signature = hashlib.md5(
        repr((fmt, path, modified, positions, teams, search)).encode(),
        usedforsecurity=False,
    ).hexdigest()[:10]
    return f"grid_{version}_{signature}"


def filter_frame(
    frame: pd.DataFrame, positions: list[str], teams: list[str], search: str
) -> pd.DataFrame:
    """Rows matching every active filter (an empty filter matches all)."""
    mask = pd.Series(True, index=frame.index)
    if positions:
        mask &= frame["Position"].isin(positions)
    if teams:
        mask &= frame["Team"].isin(teams)
    if search.strip():
        mask &= frame["Player"].str.lower().str.contains(search.strip().lower(), regex=False)
    return frame[mask]


def conflicts(lock_slot: str | None, exclude_slot: str | None) -> bool:
    """
    Whether a lock and an exclude on one player contradict each other. An
    exclude from every slot, or from the very slot locked, does; a Showdown
    lock at Any with an exclude at CPT just means "FLEX".
    """
    if not lock_slot or not exclude_slot:
        return False
    return exclude_slot == SLOT_ANY or lock_slot == exclude_slot


def _cell_slot(fmt: str, value: Any) -> str | None:
    """A grid cell's value as a stored slot (None = not selected)."""
    if fmt == CLASSIC:
        return SLOT_ANY if bool(value) else None
    return value if value in SHOWDOWN_SLOTS else None


@dataclass
class EditOutcome:
    """What applying grid edits changed."""

    changed: bool = False
    reset_grid: bool = False  # the grid must be rebuilt to show a resolved conflict
    notices: list[str] = field(default_factory=list)


def _fold_value_edits(
    fmt: str,
    changes: dict[str, Any],
    file_row: pd.Series,
    key: Any,
    name: str,
    edits: Edits,
    outcome: EditOutcome,
) -> None:
    """
    Folds one row's Projection / Ceiling / ownership cells into `edits`. A
    cell cleared to blank, or set back to the file's number, drops the edit.
    Like the lock cells, only a value that differs from the stored state is
    new, so an older pending value is never applied twice.
    """
    columns = EDITABLE_COLUMNS[fmt]
    for grid_column, value in changes.items():
        if grid_column not in columns:
            continue
        file_value = float(file_row[columns[grid_column]])
        stored = edits.get(key, {})
        current = stored.get(grid_column, file_value)
        if value is None or pd.isna(value):
            if grid_column not in stored:
                continue
            new = None
            # The blank stays pending in edited_rows and would keep drawing over
            # the restored file value; redraw so the cell shows the number used.
            outcome.reset_grid = True
        else:
            new = float(value)
            if abs(new - current) <= EDIT_TOLERANCE:
                continue
            if abs(new - file_value) <= EDIT_TOLERANCE:
                new = None
        outcome.changed = True
        if new is None:
            stored.pop(grid_column, None)
            if not stored:
                edits.pop(key, None)
        else:
            edits.setdefault(key, {})[grid_column] = new
        if fmt == SHOWDOWN and grid_column in CAPTAIN_FROM_FLEX:
            what = grid_column.lower()
            outcome.notices.append(
                f"{name}: Captain {what} recalculated at {showdown.CAPTAIN_MULTIPLIER}x "
                f"= {new * showdown.CAPTAIN_MULTIPLIER:.2f}."
                if new is not None
                else f"{name}: {what} is back to the file's value, and so is its Captain {what}."
            )


def apply_grid_edits(
    fmt: str,
    shown: pd.DataFrame,
    edited_rows: dict[Any, dict[str, Any]],
    df: pd.DataFrame,
    locks: Selections,
    excludes: Selections,
    edits: Edits | None = None,
) -> EditOutcome:
    """
    Folds the grid's edits (Streamlit's edited_rows: row position in `shown`
    -> {column: value}) into `locks` / `excludes` / `edits`, in place. `df`
    is the frame as loaded, so it supplies the file's values.

    A lock and an exclude on the same player that contradict each other are
    never both kept: the one just made wins and the older one is dropped,
    with a notice.
    """
    outcome = EditOutcome()
    for position, changes in edited_rows.items():
        row = shown.index[int(position)]
        key = player_key(fmt, df.loc[row])
        name = shown.at[row, "Player"]
        if edits is not None:
            _fold_value_edits(fmt, changes, df.loc[row], key, name, edits, outcome)
        # edited_rows keeps every edit since the grid was drawn, so a row can
        # carry an older, already-applied value in one column beside the new
        # click in the other. Only cells that differ from the stored state
        # are new; deciding that up front keeps the older one from being
        # re-applied after a conflict drops it.
        fresh = {
            column: _cell_slot(fmt, value)
            for column, value in changes.items()
            if column in (LOCK, EXCLUDE)
            and _cell_slot(fmt, value) != (locks if column == LOCK else excludes).get(key)
        }
        for column, target, other, other_name in (
            (LOCK, locks, excludes, "exclude"),
            (EXCLUDE, excludes, locks, "lock"),
        ):
            if column not in fresh:
                continue
            slot = fresh[column]
            outcome.changed = True
            if slot is None:
                target.pop(key, None)
                continue
            target[key] = slot
            pair = (slot, other.get(key)) if column == LOCK else (other.get(key), slot)
            if conflicts(*pair):
                dropped = other.pop(key)
                outcome.reset_grid = True
                where = "" if fmt == CLASSIC else f" ({dropped})"
                outcome.notices.append(
                    f"{name}: removed the {other_name}{where}; a player can't be "
                    f"locked and excluded at once."
                )
    return outcome


def describe_edits(fmt: str, df: pd.DataFrame, edits: Edits) -> list[str]:
    """ "Name: Projection 23.97 -> 30.00" for each edit present in `df`, in file order."""
    columns = EDITABLE_COLUMNS[fmt]
    lines = []
    seen = set()
    for row, key in player_keys(fmt, df).items():
        if key not in edits or key in seen:
            continue
        seen.add(key)
        parts = []
        for grid_column, value in edits[key].items():
            unit = "%" if grid_column in OWNERSHIP_GRID_COLUMNS else ""
            file_value = float(df.at[row, columns[grid_column]])
            parts.append(f"{grid_column} {file_value:.2f}{unit} -> {value:.2f}{unit}")
        lines.append(f"{str(df.at[row, 'Player']).strip()}: {'; '.join(parts)}")
    return lines


def prune_edits(fmt: str, df: pd.DataFrame, edits: Edits) -> bool:
    """
    Drops edits that now equal the file's value -- after a re-download that
    caught up with them -- in place. Returns whether anything was dropped.
    """
    columns = EDITABLE_COLUMNS[fmt]
    dropped = False
    for row, key in player_keys(fmt, df).items():
        stored = edits.get(key)
        if not stored:
            continue
        for grid_column in list(stored):
            if abs(stored[grid_column] - float(df.at[row, columns[grid_column]])) <= EDIT_TOLERANCE:
                del stored[grid_column]
                dropped = True
        if not stored:
            del edits[key]
    return dropped


def edited_keys(fmt: str, df: pd.DataFrame, edits: Edits) -> set:
    """Keys of the players in `df` that carry an edit."""
    return set(edits) & set(player_keys(fmt, df))


def describe_selections(fmt: str, df: pd.DataFrame, selections: Selections) -> list[str]:
    """Display names for the selections present in `df`, in file order."""
    labels = []
    seen = set()
    for row, key in player_keys(fmt, df).items():
        if key in selections and key not in seen:
            seen.add(key)
            slot = selections[key]
            name = str(df.at[row, "Player"]).strip()
            labels.append(name if slot == SLOT_ANY else f"{name} ({slot})")
    return labels


def missing_selections(fmt: str, df: pd.DataFrame, selections: Selections) -> int:
    """How many stored selections name players not in this file (they are ignored)."""
    return len(set(selections) - set(player_keys(fmt, df)))


# --- Settings and options ---


@dataclass
class Settings:
    """The sidebar's values, one field per CLI flag the app exposes."""

    num_lineups: int = 1
    min_uniques: int = 1
    stack: int = 0
    stack_rb: bool = False
    max_te: int | None = None
    no_dst_opp: bool = False
    min_salary: int = 0
    max_salary: int = SALARY_CAP
    target: str = TARGET_PROJECTION
    ownership_field: str = classic.OWNERSHIP_LARGE_FIELD
    export: bool = False
    dk_entries: str | None = None


def _selected_rows(fmt: str, df: pd.DataFrame, selections: Selections) -> dict[Any, list[Any]]:
    """Selection key -> the df rows carrying it, in file order."""
    rows: dict[Any, list[Any]] = {}
    for row, key in player_keys(fmt, df).items():
        if key in selections:
            rows.setdefault(key, []).append(row)
    return rows


def build_options(
    fmt: str, settings: Settings, df: pd.DataFrame, locks: Selections, excludes: Selections
) -> Any:
    """ClassicOptions or ShowdownOptions for the current settings and selections."""
    if fmt == CLASSIC:
        lock_rows = _selected_rows(fmt, df, locks)
        exclude_rows = _selected_rows(fmt, df, excludes)
        return classic.ClassicOptions(
            num_lineups=settings.num_lineups,
            min_uniques=settings.min_uniques,
            lock_ids=tuple(lock_rows),
            exclude_ids=tuple(exclude_rows),
            stack=settings.stack,
            stack_rb=settings.stack_rb,
            max_te=settings.max_te,
            no_dst_opp=settings.no_dst_opp,
            min_salary=settings.min_salary,
            target=settings.target,
            ownership_field=settings.ownership_field,
            export=settings.export,
            dk_entries=settings.dk_entries,
        )

    def selections(chosen: Selections) -> tuple[showdown.SlotSelection, ...]:
        return tuple(
            showdown.SlotSelection(tuple(rows), None if chosen[key] == SLOT_ANY else chosen[key])
            for key, rows in _selected_rows(fmt, df, chosen).items()
        )

    return showdown.ShowdownOptions(
        num_lineups=settings.num_lineups,
        min_uniques=settings.min_uniques,
        locks=selections(locks),
        excludes=selections(excludes),
        max_salary=settings.max_salary,
        min_salary=settings.min_salary,
        target=settings.target,
        export=settings.export,
        dk_entries=settings.dk_entries,
    )


def validation_errors(fmt: str, settings: Settings) -> list[str]:
    """The checks the CLI makes (strict for Classic), as messages to show inline."""
    errors = []
    try:
        if fmt == CLASSIC:
            classic.ClassicOptions(
                num_lineups=settings.num_lineups,
                min_uniques=settings.min_uniques,
                stack=settings.stack,
                max_te=settings.max_te,
                min_salary=settings.min_salary,
                target=settings.target,
                ownership_field=settings.ownership_field,
            ).validate(strict=True)
        else:
            showdown.ShowdownOptions(
                num_lineups=settings.num_lineups,
                min_uniques=settings.min_uniques,
                max_salary=settings.max_salary,
                min_salary=settings.min_salary,
                target=settings.target,
            ).validate()
    except ValueError as exc:
        errors.append(str(exc))
    # Showdown reads the entries file only to export, as the CLI does.
    needs_entries = fmt == CLASSIC or settings.export
    if needs_entries and settings.dk_entries and not os.path.isfile(settings.dk_entries):
        errors.append(f"DraftKings entries file not found: {settings.dk_entries}")
    return errors


# --- Running ---


@dataclass
class RunOutcome:
    """One Optimize click: the result (or the error that stopped it) and notes."""

    fmt: str
    target: str
    slate: str = classic.SLATE_MAIN
    result: Any = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    export_path: str | None = None
    edited: set = field(default_factory=set)  # keys of players run with edited values


def optimize(
    fmt: str,
    pool_df: pd.DataFrame,
    projections_path: str,
    settings: Settings,
    locks: Selections,
    excludes: Selections,
    edits: Edits | None = None,
) -> RunOutcome:
    """
    Runs the optimizer the way the CLI does -- Classic seats the FLEX by
    kickoff from the entries file -- on the loaded frame with `edits` laid
    over it, and exports when asked. Never raises: any failure becomes
    RunOutcome.error.
    """
    slate = classic.detect_slate(projections_path) if fmt == CLASSIC else classic.SLATE_MAIN
    outcome = RunOutcome(fmt, settings.target, slate)
    notes: list[str] = []
    try:
        df = apply_overrides(fmt, pool_df, edits)
        outcome.edited = edited_keys(fmt, pool_df, edits or {})
        if outcome.edited:
            notes.append(f"Optimized with edited values for {len(outcome.edited)} player(s).")
        if fmt == CLASSIC:
            if settings.dk_entries is None:
                notes.append("No DraftKings entries file selected; the FLEX is not reordered by kickoff.")
                kickoffs = {}
            else:
                kickoffs = classic.load_dk_kickoffs(settings.dk_entries, notes)
            df = classic.attach_kickoffs(df, kickoffs, notes)
        options = build_options(fmt, settings, df, locks, excludes)
        module = classic if fmt == CLASSIC else showdown
        result = module.run(df, options)
        outcome.result = result
        if settings.export:
            lookup = load_dk_name_ids(settings.dk_entries)
            if lookup is None:
                notes.append(missing_upload_rows_note(settings.dk_entries))
            rows = module.export_rows(result, lookup, notes)
            if fmt == CLASSIC:
                outcome.export_path = classic.write_export(rows, slate, settings.target)
            else:
                outcome.export_path = showdown.write_export(rows, settings.target)
    except (ValueError, FileNotFoundError, OSError) as exc:
        outcome.error = str(exc)
    except Exception as exc:  # noqa: BLE001 -- the app shows every failure, never a traceback
        outcome.error = f"Unexpected error: {type(exc).__name__}: {exc}"
    outcome.notes = [note.strip() for note in notes]
    return outcome


# --- Results ---


EDITED_MARK = " ✎"


def lineup_table(
    fmt: str, lineup: Any, show_kickoff: bool = False, edited: set | None = None
) -> pd.DataFrame:
    """
    One lineup as a display table, in the CLI's slot order. Players whose key
    is in `edited` get a pencil after their name.
    """
    edited = edited or set()

    def name(player: Any) -> str:
        mark = EDITED_MARK if player_key(fmt, player) in edited else ""
        return str(player["Player"]).strip() + mark

    if fmt == SHOWDOWN:
        return pd.DataFrame(
            [
                {
                    "Slot": r["Slot"],
                    "Player": name(r),
                    "Pos": r["Position"],
                    "Team": r["Team"],
                    "Salary": r["Salary"],
                    "Proj": r["Projection"],
                    "Own%": r["Ownership"],
                    "Ceiling": r["Ceiling"],
                }
                for r in lineup.rows
            ]
        )
    rows = []
    for slot, player in lineup.slot_rows():
        if player is None:
            continue
        row = {
            "Slot": classic.display_slot(slot),
            "Player": name(player),
            "Pos": player["Position"],
            "Team": player["Team"],
            "Salary": int(player["Salary"]),
            "Proj": player["Projection"],
            "Own%": player["Ownership"],
            "Ceiling": player["Ceiling"],
        }
        if show_kickoff:
            row["Kickoff (ET)"] = classic.format_kickoff(player.get("Kickoff"))
        rows.append(row)
    return pd.DataFrame(rows)


def lineup_totals(fmt: str, lineup: Any, target: str) -> str:
    """The CLI's per-lineup totals on one line."""
    parts = []
    if target == TARGET_BLEND:
        parts.append(f"Blend score {target_score(target, lineup.projection, lineup.ceiling):.2f}")
    parts += [
        f"Projection {lineup.projection:.2f}",
        f"Ceiling {lineup.ceiling:.2f}",
        f"Ownership {lineup.ownership:.2f}%",
    ]
    salary = f"Salary ${lineup.salary:,}"
    if fmt == SHOWDOWN:
        salary += f" (${SALARY_CAP - lineup.salary:,} remaining)"
    parts.append(salary)
    return " · ".join(parts)

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

import argparse
import traceback
from collections.abc import Sequence

import pandas as pd

from nfl_dfs_optimizer import common
from nfl_dfs_optimizer.cli._shared import print_notes, reject_stray_projections_path
from nfl_dfs_optimizer.common import (
    DK_ENTRIES_GLOB,
    OPTIMIZATION_TARGETS,
    SALARY_CAP,
    TARGET_BLEND,
    TARGET_PROJECTION,
    PlayerDataError,
    check_target_data,
    load_dk_name_ids,
    missing_upload_rows_note,
    resolve_optimization_target,
)
from nfl_dfs_optimizer.showdown import (
    PROJECTIONS_GLOB,
    VALID_SLOTS,
    ShowdownLineup,
    ShowdownOptions,
    SlotSelection,
    duplicate_player_groups,
    lineup_export_rows,
    load_player_data,
    run,
    write_export,
)

TABLE_WIDTH: int = 91


def build_parser() -> argparse.ArgumentParser:
    """The Showdown CLI's argument parser."""
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
    return parser


def parse_player_selector(token: str) -> tuple[str, str | None]:
    """
    Splits a -l / -x argument into a player name and an optional slot qualifier.

    Accepts a bare name ("Drake Maye") meaning "any slot", or a name with a
    trailing ":CPT" / ":FLEX" suffix to target one specific roster slot.
    """
    if ":" in token:
        name, _, suffix = token.rpartition(":")
        if name.strip() and suffix.strip().upper() in VALID_SLOTS:
            return name.strip(), suffix.strip().upper()
    return token.strip(), None


def _dedupe_selectors(tokens: Sequence[str]) -> list[tuple[str, str | None]]:
    """Parses -l / -x tokens into (name, slot) pairs, dropping exact repeats."""
    seen = set()
    parsed: list[tuple[str, str | None]] = []
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


def resolve_selectors(
    players_df: pd.DataFrame, tokens: list[str], verb: str, action: str
) -> list[SlotSelection]:
    """
    Turns -l / -x tokens ("Name", "Name:CPT") into row selections, printing
    each match and each miss the way the script always has.
    """
    selections: list[SlotSelection] = []
    for player_name, slot in _dedupe_selectors(tokens):
        matches = _find_player_indices(players_df, player_name)
        if matches.empty:
            print(
                f"  WARNING: Player '{player_name}' not found in projections. "
                f"Skipping {action}."
            )
            continue
        selections.append(SlotSelection(tuple(matches), slot))
        shown = matches[:1] if verb == "Locked" else matches
        for idx in shown:
            print(f"  {verb}: {players_df.loc[idx, 'Player']} @ {slot or 'ANY SLOT'}")
    return selections


def print_lineup(lineup: ShowdownLineup, target: str = TARGET_PROJECTION) -> None:
    """Prints a single Showdown lineup in a human-readable table."""
    print(f"\n--- Optimal NFL Showdown Lineup #{lineup.number} ---")
    if target == TARGET_BLEND:
        # A blended run's score matches neither printed total.
        print(f"Blend Score ({OPTIMIZATION_TARGETS[target][0]}): {lineup.score:.2f}")
    print(f"Projection: {lineup.projection:.2f}")
    print(f"Total Ownership: {lineup.ownership:.2f}%")
    print(f"Ceiling: {lineup.ceiling:.2f}")
    print(f"Salary: ${lineup.salary:,} (${SALARY_CAP - lineup.salary:,} remaining)")
    print("-" * TABLE_WIDTH)
    print(
        f"{'Slot':<6} {'Player':<25} {'Pos':<5} {'Team':<6} "
        f"{'Salary':>8} {'Proj':>8} {'Own%':>8} {'Ceiling':>9}"
    )
    print("-" * TABLE_WIDTH)
    for row in lineup.rows:
        print(
            f"{row['Slot']:<6} {row['Player']!s:<25} {row['Position']!s:<5} "
            f"{row['Team']!s:<6} ${row['Salary']:>7,} "
            f"{row['Projection']:>8.2f} {row['Ownership']:>7.2f}% "
            f"{row['Ceiling']:>9.2f}"
        )
    print("-" * TABLE_WIDTH)


def main() -> None:
    """Parses the command line, runs the optimizer, and prints the lineups."""
    args = build_parser().parse_args()

    try:
        # --- 1. Validate arguments ---
        target = resolve_optimization_target(args.ceiling, args.projceiling)
        base_options = ShowdownOptions(
            num_lineups=args.num_lineups,
            min_uniques=args.min_uniques,
            max_salary=args.max_salary,
            min_salary=args.min_salary,
            target=target,
            export=args.export,
            dk_entries=args.dk_entries,
        )
        notes: list[str] = []
        try:
            base_options.validate(notes)
        finally:
            print_notes(notes)

        # --- 2. Load and prepare data ---
        reject_stray_projections_path(args.filepath, args.lock, args.exclude)
        notes = []
        try:
            projections_path = common.find_projections_file(
                PROJECTIONS_GLOB, args.filepath, notes=notes
            )
        finally:
            print_notes(notes)
        print(
            f"Projections file: {projections_path} "
            f"({common.describe_modified(projections_path)})"
        )
        try:
            pool = load_player_data(projections_path)
        except PlayerDataError as exc:
            print_notes(exc.notes)
            raise
        print_notes(pool.notes)
        players_df = pool.df
        print(f"Optimizing on: {OPTIMIZATION_TARGETS[target][0]}.")
        warning = check_target_data(players_df, target)
        if warning:
            print(warning)

        # --- 3. Rules, echoed as the model is built ---
        if args.min_salary > 0:
            print(f"\nEnforcing a minimum lineup salary of ${args.min_salary:,}...")
        duplicate_groups = duplicate_player_groups(players_df)
        if duplicate_groups:
            print(
                f"\nNOTE: {len(duplicate_groups)} player(s) appear on multiple rows. "
                f"Constraining each to at most one roster spot:"
            )
            for (name, team), idxs in duplicate_groups.items():
                print(f"  {name} ({team}) - {len(idxs)} rows")
        locks: list[SlotSelection] = []
        if args.lock:
            print(f"\nLocking players: {args.lock}")
            locks = resolve_selectors(players_df, args.lock, "Locked", "lock")
        excludes: list[SlotSelection] = []
        if args.exclude:
            print(f"\nExcluding players: {args.exclude}")
            excludes = resolve_selectors(players_df, args.exclude, "Excluded", "exclusion")

        # DraftKings "Name + ID" values for the upload row that follows each
        # lineup's totals. Absent or unreadable entries file: no upload rows.
        entries_path = None
        if args.export:
            notes = []
            try:
                entries_path = common.find_dk_entries_file(args.dk_entries, notes=notes)
            finally:
                print_notes(notes)
        dk_lookup = load_dk_name_ids(entries_path) if args.export else None
        if args.export and dk_lookup is None:
            print(missing_upload_rows_note(entries_path))

        # --- 4. Optimize ---
        options = ShowdownOptions(
            num_lineups=args.num_lineups,
            min_uniques=args.min_uniques,
            locks=tuple(locks),
            excludes=tuple(excludes),
            max_salary=args.max_salary,
            min_salary=args.min_salary,
            target=target,
            export=args.export,
            dk_entries=args.dk_entries,
        )
        result = run(players_df, options)

        export_data = []
        for lineup in result.lineups:
            print(f"\n--- Generating Lineup #{lineup.number} ---")
            print_lineup(lineup, target)
            if args.export:
                notes = []
                export_data.extend(lineup_export_rows(lineup, dk_lookup, notes))
                print_notes(notes)
        if not result.complete:
            print(f"\n--- Generating Lineup #{len(result.lineups) + 1} ---")
            print_notes(result.messages)

        # --- 5. Export All Lineups to CSV ---
        if args.export and export_data:
            path = write_export(export_data, target)
            print(f"\nAll generated lineups exported to: {path}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nFATAL ERROR: {e}")
    except Exception as e:  # noqa: BLE001 -- the CLI reports anything unexpected
        print(f"\nAn unexpected error occurred: {type(e).__name__}: {e}")
        # Send the traceback to stderr so stdout stays clean for the lineups
        # while a bug report still carries something actionable.
        traceback.print_exc()


if __name__ == "__main__":
    main()

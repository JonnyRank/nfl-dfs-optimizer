"""
DraftKings NFL Multi-Lineup Optimizer.

This script ingests a CSV file with player projections and uses mixed integer
linear programming (PuLP + HiGHS) to find a specified number of unique, optimal
lineups that maximize a chosen scoring target (projection, ceiling, or a 50/50 mix of the
two), subject to DraftKings' classic NFL contest rules.

The script is run from the command line. It optimizes on the newest
"DraftKings NFL DFS Projections*.csv" in your Downloads folder, or on the
projections CSV whose path is given as the first argument.

Input Arguments:
    python NFL-Multi-Opto-v2.0.py ["path"] -n -u -e -te -x -l -s -srb -ndo -mns -c -pj -sf -ls -dk
    python <script> <proj file (optional)> <# of lineups> <min uniques> <max TE> <exclude> <export to CSV> <lock players> <stack QB with WR/TE> <stack QB with RB> <no DST vs Opp> <minimum salary> <optimize on ceiling> <optimize on 50/50 proj+ceiling> <use small-field ownership> <late swap DKEntries.csv> <DKEntries file path>
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -e -l "Josh Allen" -s -ndo
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -c
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -pj
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -mns 49500
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -ls -u 2 -mns 49500
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -e -dk "C:\\path\\to\\DKEntries.csv"
    python NFL-Multi-Opto-v2.0.py -n 5 -u 2 -e

Projections File (optional first argument):
    Omitted, the newest "DraftKings NFL DFS Projections*.csv" in Downloads is
    used -- Main, Early, and Late slate downloads and their "(1)" re-downloads
    all match. Given, it must be the path to a file (a folder is rejected). The
    slate is read from the file name: "Early Slate" and "Late Slate" label
    each printed lineup and add "_early" / "_late" after "nfl_classic" in the
    export file name; "Main Slate", or no slate in the name, is Main and keeps
    the usual file name. Export contents do not change.

DraftKings Entries File (-dk / --dk-entries):
    One entries file serves the whole run: the export's upload row, the
    kickoffs that seat the FLEX, and the entries late swap rebuilds. By default
    it is the newest DKEntries*.csv in your Downloads folder; -dk names another,
    and a -dk path that does not exist stops the run.

Late Swap (-ls / --late-swap):
    Reads the DraftKings entries file (see -dk above) and re-optimizes
    every entry in it. A player whose game has started -- judged from Game
    Info against the clock at runtime, or DraftKings' own LOCKED tag -- stays
    in his slot. Every other slot, including unstarted players already in the
    lineup, is refilled from players whose games have not started. -u is
    enforced between entries of the same contest only. All entries are written
    to Downloads\\upload-ready-DKEntries-<timestamp>.csv (with "-early" or
    "-late" before the timestamp for those slates), each cell holding
    DraftKings' own "Name + ID" text. -l, -x, -s, -srb, -te, -ndo, -mns, -c,
    -pj and -sf apply; -n and -e do not.

Minimum Salary (-mns / --min-salary):
    Requires each lineup to spend at least the given total salary; omitted,
    there is no floor. With --late-swap the floor covers the whole entry, so
    the locked players' salary counts toward it, and an entry that cannot
    reach it is rebuilt without the floor rather than left unchanged.

Optimization Targets:
    (default)               Maximize total projection.
    -c / --ceiling          Maximize total ceiling.
    -pj / --projceiling     Maximize an equally weighted 50/50 blend of the two.
    The two flags are mutually exclusive; omitting both keeps the historical
    projection-only behavior.

Input Columns:
    Headers are resolved through COLUMN_ALIASES, so both the legacy and the
    current projections headers load without editing the CSV:
        ID          <- "ID", "id", "DK ID", or "Player ID"
        Player      <- "Player", "Name", or "Player Name"
        Position    <- "Position", "DK Pos", or "Pos"
        Team        <- "Team" or "Tm"
        Opp         <- "Opp" or "Opponent"
        Salary      <- "Salary" or "DK Salary"
        Projection  <- "Projection", "Proj", or "DK Proj"
        Ceiling     <- "Ceiling" or "DK Ceiling"      (optional)
        Ownership   <- "Large Field", "Ownership", or "Own"   (optional;
                       -sf / --small-field takes "Small Field" only, never an
                       unlabeled legacy column that may hold the other field)
    A header matching nothing in the table falls back to fuzzy matching and is
    reported when it resolves; a required column that stays unresolved raises
    with the headers the file actually contained.

Key Features:
- Loads player data from a command-line specified CSV file.
- Accepts either the legacy or the current projections headers.
- Cleans and validates player salary, projection, and ownership data.
- Identifies unique games to enforce the "at least two games" rule.
- Uses the HiGHS solver (via highspy) through PuLP to solve each lineup.
- Optimizes on projection, ceiling, or a 50/50 blend of the two.
- Enforces constraints for salary cap, an optional salary floor, roster
  composition (QB, RB, WR, TE, FLEX, DST), and lineup diversity.
- Prints a well-formatted, human-readable optimal lineup.
- Seats the latest-kickoff player in FLEX, using the Game Info times in the
  DraftKings entries file (skipped when there is none).

FLEX Seating:
    The solver picks nine players; which one prints as FLEX is decided
    afterward. The position counts fix which position supplies the FLEX (a
    third RB means an RB sits there), and within that position the player
    whose game kicks off latest takes the FLEX, keeping the most late-swap
    options open. Kickoffs come from the Game Info column of the entries file,
    matched by player ID, and print in a "Kickoff (ET)" column. A player whose
    game shows "In Progress" has no kickoff, prints "-", and counts as the
    earliest. With no entries file the FLEX falls back to the cheapest player
    of that position. The export CSV and its upload row follow the same seating.
"""

import argparse

import pandas as pd

from nfl_dfs_optimizer import common
from nfl_dfs_optimizer.classic import (
    DISPLAY_ORDER,
    OWNERSHIP_LARGE_FIELD,
    OWNERSHIP_SMALL_FIELD,
    PROJECTIONS_GLOB,
    ClassicLineup,
    ClassicOptions,
    attach_kickoffs,
    detect_slate,
    display_slot,
    format_kickoff,
    lineup_export_rows,
    load_dk_kickoffs,
    load_player_data,
    run,
    stop_messages,
    with_ownership,
    write_export,
)
from nfl_dfs_optimizer.cli._shared import print_notes, reject_stray_projections_path
from nfl_dfs_optimizer.common import (
    DK_ENTRIES_GLOB,
    OPTIMIZATION_TARGETS,
    TARGET_BLEND,
    PlayerDataError,
    check_target_data,
    load_dk_name_ids,
    missing_upload_rows_note,
    resolve_optimization_target,
)
from nfl_dfs_optimizer.late_swap import run_late_swap


def build_parser() -> argparse.ArgumentParser:
    """The Classic CLI's argument parser."""
    parser = argparse.ArgumentParser(
        description="DraftKings NFL Multi-Lineup Optimizer."
    )
    parser.add_argument(
        "filepath",
        type=str,
        nargs="?",
        help=(
            f"Path to the DraftKings projections CSV file. Omit it to use the "
            f"newest {PROJECTIONS_GLOB} in Downloads."
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
        help="Minimum number of unique players between lineups (default: 1).",
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
        help="List of player names to lock into the lineup (case-insensitive).",
    )
    parser.add_argument(
        "-ndo",
        "--no-dst-opp",
        action="store_true",
        help="Disallow selecting a DST and any offensive player from their opponent.",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        help="List of player names to exclude from the lineup (case-insensitive).",
    )
    parser.add_argument(
        "-s",
        "--stack",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Stack QB with at least N WR/TEs from the same team (default: 1 if flag used).",
    )
    parser.add_argument(
        "-srb",
        "--stack-rb",
        action="store_true",
        help="Stack QB with at least one RB from the same team.",
    )
    parser.add_argument(
        "-te",
        "--max-te",
        type=int,
        help="Maximum number of TEs allowed in a lineup (e.g., 1 to ban TE in FLEX).",
    )
    parser.add_argument(
        "-mns",
        "--min-salary",
        type=int,
        default=0,
        help=(
            "Minimum total lineup salary (default: no floor). Applies to "
            "--late-swap too, where locked players count toward the floor."
        ),
    )
    parser.add_argument(
        "-sf",
        "--small-field",
        action="store_true",
        help=(
            "Display small-field ownership instead of large-field ownership "
            "(display/export only; ownership is not optimized on)."
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
        "-ls",
        "--late-swap",
        action="store_true",
        help=(
            "Late-swap the entries in the newest DKEntries*.csv in Downloads: "
            "players whose games have started stay put, every other slot is "
            "re-optimized, and upload-ready-DKEntries-<timestamp>.csv is written "
            "to Downloads. -u applies within each contest; -n and -e are ignored."
        ),
    )
    parser.add_argument(
        "-dk",
        "--dk-entries",
        metavar="PATH",
        help=(
            f"DraftKings entries CSV to read instead of the newest "
            f"{DK_ENTRIES_GLOB} in Downloads. It supplies the export's upload "
            f"row, the kickoffs that seat the FLEX, and the entries -ls swaps."
        ),
    )
    return parser


def resolve_player_names(
    players_df: pd.DataFrame, names: list[str] | None, verb: str, action: str
) -> list[int]:
    """
    Matches -l / -x names (case-insensitive, exact) to player IDs, printing
    each match as "<verb>: Name (ID: ...)" and each miss as a warning.
    """
    ids: list[int] = []
    for player_name in names or []:
        matches = players_df[players_df["Player"].str.lower() == player_name.lower()]
        if matches.empty:
            print(f"  WARNING: Player '{player_name}' not found in projections. Skipping {action}.")
            continue
        for idx in matches.index:
            ids.append(int(players_df.loc[idx, "ID"]))
            print(f"  {verb}: {players_df.loc[idx, 'Player']} (ID: {players_df.loc[idx, 'ID']})")
    return ids


def print_lineup(lineup: ClassicLineup, slate: str, target: str, show_kickoff: bool) -> None:
    """Prints one lineup's totals and its slot table."""
    table_rule = "-" * (104 if show_kickoff else 90)
    print(f"\n--- Optimal NFL {slate} Slate Lineup #{lineup.number} ---")
    # A blended run's score matches neither printed total, so show it; for the
    # other targets the Projection/Ceiling lines already are it.
    if target == TARGET_BLEND:
        print(f"Blend Score ({OPTIMIZATION_TARGETS[target][0]}): {lineup.score:.2f}")
    print(f"Projection: {lineup.projection:.2f}")
    print(f"Ownership: {lineup.ownership:.2f}%")
    print(f"Ceiling: {lineup.ceiling:.2f}")
    print(f"Salary: ${lineup.salary:,}")
    print(table_rule)
    print(
        f"{'Slot':<5} {'Player':<25} {'Pos':<5} {'Team':<5} "
        f"{'Salary':>8} {'Proj':>8} {'Own%':>8} {'Ceiling':>8}"
        + (f"  {'Kickoff (ET)':>12}" if show_kickoff else "")
    )
    print(table_rule)
    for slot in DISPLAY_ORDER:
        player = lineup.players.get(slot)
        if player:
            print(
                f"{display_slot(slot):<5} {player['Player']:<25} {player['Position']:<5} "
                f"{player['Team']:<5} ${int(player['Salary']):>7,} "
                f"{player['Projection']:>8.2f} {player['Ownership']:>7.2f}% "
                f"{player['Ceiling']:>8.2f}"
                + (f"  {format_kickoff(player['Kickoff']):>12}" if show_kickoff else "")
            )
        else:
            print(f"{slot:<5} - ERROR ASSIGNING PLAYER -")
    print(table_rule)


def main() -> None:
    """Parses the command line, runs the optimizer, and prints the lineups."""
    args = build_parser().parse_args()

    try:
        target = resolve_optimization_target(args.ceiling, args.projceiling)
        ownership_field = OWNERSHIP_SMALL_FIELD if args.small_field else OWNERSHIP_LARGE_FIELD
        # Argument validation, before anything is read from disk.
        ClassicOptions(
            min_salary=args.min_salary, target=target, ownership_field=ownership_field
        ).validate()

        reject_stray_projections_path(args.filepath, args.lock, args.exclude)
        notes: list[str] = []
        try:
            projections_path = common.find_projections_file(
                PROJECTIONS_GLOB, args.filepath, notes=notes
            )
        finally:
            print_notes(notes)
        slate = detect_slate(projections_path)
        print(
            f"Projections file: {projections_path} "
            f"({slate} Slate, {common.describe_modified(projections_path)})"
        )
        try:
            pool = load_player_data(projections_path)
        except PlayerDataError as exc:
            print_notes(exc.notes[ownership_field])
            raise
        print_notes(pool.notes[ownership_field])
        print(f"Optimizing on: {OPTIMIZATION_TARGETS[target][0]}.")
        warning = check_target_data(pool.df, target)
        if warning:
            print(warning)
        players_df = with_ownership(pool.df, ownership_field)

        if args.late_swap:
            run_late_swap(args, players_df, target, slate=slate)
            return

        # One entries file serves the run: FLEX kickoffs and the upload row.
        notes = []
        try:
            entries_path = common.find_dk_entries_file(args.dk_entries, notes=notes)
        finally:
            print_notes(notes)
        notes = []
        kickoffs = load_dk_kickoffs(entries_path, notes)
        players_df = attach_kickoffs(players_df, kickoffs, notes)
        print_notes(notes)

        if args.min_salary > 0:
            print(f"\nEnforcing a minimum lineup salary of ${args.min_salary:,}...")
        lock_ids: list[int] = []
        if args.lock:
            print(f"\nLocking players: {args.lock}")
            lock_ids = resolve_player_names(players_df, args.lock, "Locked", "lock")
        exclude_ids: list[int] = []
        if args.exclude:
            print(f"\nExcluding players: {args.exclude}")
            exclude_ids = resolve_player_names(players_df, args.exclude, "Excluded", "exclusion")
        if args.stack > 0:
            print(f"\nEnforcing 'QB + {args.stack} WR/TE Stack' rule...")
        if args.stack_rb:
            print("\nEnforcing 'QB + RB Stack' rule...")
        if args.no_dst_opp:
            print("\nEnforcing 'No DST vs Opponent' rule...")
        if args.max_te is not None:
            print(f"\nEnforcing maximum of {args.max_te} TE(s)...")

        options = ClassicOptions(
            num_lineups=args.num_lineups,
            min_uniques=args.min_uniques,
            lock_ids=tuple(lock_ids),
            exclude_ids=tuple(exclude_ids),
            stack=args.stack,
            stack_rb=args.stack_rb,
            max_te=args.max_te,
            no_dst_opp=args.no_dst_opp,
            min_salary=args.min_salary,
            target=target,
            ownership_field=ownership_field,
            export=args.export,
            dk_entries=args.dk_entries,
        )
        result = run(players_df, options)

        # DraftKings "Name + ID" values for the upload row that follows each
        # lineup's totals. Absent or unreadable entries file: no upload rows.
        dk_lookup = load_dk_name_ids(entries_path) if args.export else None
        if args.export and dk_lookup is None:
            print(missing_upload_rows_note(entries_path))

        export_data = []
        for lineup in result.lineups:
            print(f"\n--- Generating Lineup #{lineup.number} ---")
            print_lineup(lineup, slate, target, result.show_kickoff)
            if args.export:
                notes = []
                export_data.extend(lineup_export_rows(lineup, dk_lookup, notes))
                print_notes(notes)
        if not result.complete:
            print(f"\n--- Generating Lineup #{len(result.lineups) + 1} ---")
            print_notes(stop_messages(len(result.lineups), result.status, args.min_salary))

        if args.export and export_data:
            path = write_export(export_data, slate, target)
            print(f"\nAll generated lineups exported to: {path}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nFATAL ERROR: {e}")
    except Exception as e:  # noqa: BLE001 -- the CLI reports anything unexpected, never a traceback dump
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

"""
The Classic CLI's -ls: prints a late swap as the legacy script did, over the
late_swap core. See the "Late Swap (-ls / --late-swap)" section of
cli/classic.py for the flag's documentation.
"""

import argparse
from datetime import datetime

import pandas as pd

from nfl_dfs_optimizer import common, late_swap
from nfl_dfs_optimizer.classic import SLATE_MAIN
from nfl_dfs_optimizer.cli._shared import print_notes
from nfl_dfs_optimizer.common import DK_ENTRIES_GLOB, build_solver
from nfl_dfs_optimizer.late_swap import EntryOutcome, LateSwapOptions, LateSwapSlate


def _ids_for_names(players_df: pd.DataFrame, names: list[str] | None, action: str) -> set[int]:
    """Resolves -l / -x player names (case-insensitive) to DraftKings IDs."""
    ids: set[int] = set()
    for name in names or []:
        matches = players_df[players_df["Player"].str.lower() == name.lower()]
        if matches.empty:
            print(f"  WARNING: Player '{name}' not found in projections. Skipping {action}.")
            continue
        ids.update(int(dk_id) for dk_id in matches["ID"])
    return ids


def _print_entry(slate: LateSwapSlate, outcome: EntryOutcome, total: int) -> None:
    """Prints one late-swapped entry with a per-slot LOCKED/KEEP/MOVE/NEW status."""

    def change(label: str, before: float, after: float, unit: str = "") -> str:
        return f"{label}: {before:.2f}{unit} -> {after:.2f}{unit} ({after - before:+.2f}{unit})"

    totals = late_swap.entry_totals(slate, outcome)
    # Projection, Ceiling and Ownership always show before -> after, whatever
    # the target: points on one line, Ownership and Salary on the next.
    points = [change(column, *totals[column]) for column in ("Projection", "Ceiling")]
    ownership = change("Ownership", *totals["Ownership"], "%")
    entry = outcome.entry
    print(f"\n--- Entry {outcome.number}/{total}: {entry.entry_id} | {entry.contest_name} ---")
    print(" | ".join(points))
    print(f"{ownership} | Salary: ${late_swap.entry_salary(slate, outcome):,}")
    if outcome.note:
        print(f"NOTE: {outcome.note}")
    print("-" * 97)
    print(
        f"{'Slot':<5} {'Player':<25} {'Pos':<5} {'Team':<5} "
        f"{'Salary':>8} {'Proj':>8} {'Own%':>8} {'Ceiling':>8}  Status"
    )
    print("-" * 97)
    for idx, row in enumerate(late_swap.entry_rows(slate, outcome)):
        if row is None:
            print(f"{slate.slot_labels[idx]:<5} - EMPTY -")
            continue
        print(
            f"{row['Slot']:<5} {row['Player'][:25]:<25} {row['Pos']:<5} {row['Team']:<5} "
            f"${row['Salary']:>7,} {row['Proj']:>8.2f} "
            f"{row['Own%']:>7.2f}% {row['Ceiling']:>8.2f}  {row['Status']}"
        )
    print("-" * 97)


def run_late_swap(
    args: argparse.Namespace,
    players_df: pd.DataFrame,
    target: str,
    now: datetime | None = None,
    slate: str = SLATE_MAIN,
) -> str | None:
    """
    Late-swaps every entry in the DraftKings entries file (-dk, else the
    newest one in Downloads), printing each, and writes the upload file.

    Args:
        args: Parsed CLI options.
        players_df: Projections from load_player_data(); matched by DK ID.
        target: Optimization target key.
        now: Moment eligibility is judged at; defaults to the current time.
        slate: The projections file's slate, shown in the heading.

    Returns:
        Path of the upload file written, or None if nothing was written.
    """
    # Checked before any file is read, so a missing highspy stops the run
    # rather than surfacing once entries are already being rebuilt.
    solver = build_solver()

    notes: list[str] = []
    try:
        entries_path = common.find_dk_entries_file(args.dk_entries, notes=notes)
    finally:
        print_notes(notes)
    if entries_path is None:
        raise FileNotFoundError(
            f"No {DK_ENTRIES_GLOB} file found in {common.DOWNLOADS_DIR}. Download your "
            f"entries CSV from DraftKings' Edit Entries page first, or name one "
            f"with -dk."
        )
    swap = late_swap.load_slate(players_df, entries_path, now)
    print(f"\n--- Late Swap ({slate} Slate) ---")
    print(f"Entries file: {entries_path}")
    print(
        f"Clock: {swap.now:%m/%d/%Y %I:%M%p} ET. {len(swap.entries)} entries across "
        f"{len(swap.contests)} contest(s)."
    )
    if args.min_salary:
        print(
            f"Salary floor: ${args.min_salary:,} per entry, locked players "
            f"included; dropped for an entry that cannot reach it."
        )
    if args.num_lineups != 1 or args.export:
        print(
            "NOTE: -n and -e do not apply to --late-swap; every entry is "
            "re-optimized and written to the upload file."
        )
    swap.check_projections()

    if args.lock:
        print(f"\nLocking players: {args.lock}")
    lock_ids = _ids_for_names(players_df, args.lock, "lock")
    started_locks = swap.started_lock_names(lock_ids)
    if started_locks:
        print(f"NOTE: {late_swap.started_locks_note(started_locks)}")
    if args.exclude:
        print(f"\nExcluding players: {args.exclude}")
    exclude_ids = _ids_for_names(players_df, args.exclude, "exclusion")

    candidates = swap.candidates(exclude_ids)
    print(late_swap.pool_summary(swap, candidates))
    unprojected = swap.unprojected_names()
    if unprojected:
        print(f"NOTE: {late_swap.unprojected_note(unprojected)}")

    # The legacy script never range-checked -u / -s / -te, so these options
    # are not validated here either.
    options = LateSwapOptions(
        min_uniques=args.min_uniques,
        lock_ids=tuple(sorted(lock_ids)),
        exclude_ids=tuple(sorted(exclude_ids)),
        stack=args.stack,
        stack_rb=args.stack_rb,
        max_te=args.max_te,
        no_dst_opp=args.no_dst_opp,
        min_salary=args.min_salary,
        target=target,
    )
    outcomes = late_swap.rebuild_entries(swap, options, candidates, solver)
    for outcome in outcomes:
        _print_entry(swap, outcome, len(outcomes))

    print(f"\n{late_swap.summarize(outcomes)}")
    output_path = late_swap.write_upload(swap, outcomes, slate)
    print(f"Upload file written to: {output_path}")
    return output_path

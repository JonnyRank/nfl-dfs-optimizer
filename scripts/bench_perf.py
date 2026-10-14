"""
Timing harness for lineup generation: Classic and Showdown multi-lineup runs
and Classic late swap, timed with time.perf_counter().

    uv run python scripts/bench_perf.py               # newest real files in Downloads
    uv run python scripts/bench_perf.py --fixtures    # tests/fixtures/public/ instead
    uv run python scripts/bench_perf.py --quick       # n = 20 only, small late-swap scale

What it reports, per run:

    build    model construction, once per run: run() start to the first solve
    rebuild  per solve, the time PuLP spends creating a fresh highspy.Highs,
             re-adding every column and row, and reading the solution back
    solve    per solve, HiGHS itself (Highs.run)
    other    per lineup, everything else in the loop: extracting the lineup,
             seating FLEX, adding the diversity cut in PuLP

The split comes from TimedHiGHS, a pulp.HiGHS subclass swapped in for each
core's build_solver(), so the cores run unmodified. Late swap also runs on
synthetic entries files (the real one's entries replicated over more
contests) written to a temp folder; nothing is written to the repo,
Downloads or the export drive.
"""

import argparse
import csv
import glob
import os
import platform
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib.metadata import version

import pulp

from nfl_dfs_optimizer import classic, common, late_swap, showdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "tests", "fixtures", "public")
FIXTURE_FILES = {
    "classic": os.path.join(PUBLIC, "DraftKings NFL DFS Projections -- Main Slate.csv"),
    "showdown": os.path.join(PUBLIC, "DK NFL Showdown Projections.csv"),
    "entries": os.path.join(PUBLIC, "DKEntriesClassicLateSwap.csv"),
}

LINEUP_COUNTS = (20, 150, 500)
MIN_UNIQUES = (1, 2, 3)
# (entries, contests) for the synthetic late-swap files.
SWAP_SCALES = ((20, 4), (60, 6), (150, 10))
QUICK_SWAP_SCALES = ((20, 4),)


@dataclass
class SolveLog:
    """What TimedHiGHS saw during one run."""

    first_solve_at: float | None = None
    rebuild: list[float] = field(default_factory=list)
    solve: list[float] = field(default_factory=list)


class TimedHiGHS(pulp.HiGHS):
    """pulp.HiGHS that records the transfer and solve time of every solve."""

    log = SolveLog()

    def actualSolve(self, lp):
        log = TimedHiGHS.log
        start = time.perf_counter()
        if log.first_solve_at is None:
            log.first_solve_at = start
        self.createAndConfigureSolver(lp)
        self.buildSolverModel(lp)
        built = time.perf_counter()
        self.callSolver(lp)
        solved = time.perf_counter()
        status, sol_status = self.findSolutionValues(lp)
        for var in lp.variables():
            var.modified = False
        for constraint in lp._constraints.values():
            constraint.modifier = False
        lp.assignStatus(status, sol_status)
        done = time.perf_counter()
        log.rebuild.append((built - start) + (done - solved))
        log.solve.append(solved - built)
        return status


def timed_build_solver() -> pulp.LpSolver:
    return TimedHiGHS(msg=False)


@contextmanager
def instrumented() -> Iterator[SolveLog]:
    """Swaps TimedHiGHS into every core for the duration, yielding a fresh log."""
    modules = (classic, showdown, late_swap)
    saved = [module.build_solver for module in modules]
    TimedHiGHS.log = SolveLog()
    for module in modules:
        module.build_solver = timed_build_solver
    try:
        yield TimedHiGHS.log
    finally:
        for module, original in zip(modules, saved, strict=True):
            module.build_solver = original


# --- Inputs ---


def newest(pattern: str) -> list[str]:
    paths = [p for p in glob.glob(os.path.join(glob.escape(common.DOWNLOADS_DIR), pattern))
             if os.path.isfile(p)]
    return sorted(paths, key=os.path.getmtime, reverse=True)


def classic_entries_file(candidates: list[str]) -> str | None:
    """The first file that reads as a Classic entries file with entries filled in."""
    far_past = datetime(2000, 1, 1, tzinfo=classic._game_info_timezone())
    for path in candidates:
        try:
            _, _, entries, _ = late_swap.load_dk_entries_file(path, far_past)
        except (ValueError, FileNotFoundError):
            continue
        if any(any(cell.strip() for cell in entry.cells) for entry in entries):
            return path
    return None


def default_now(entries_path: str) -> datetime:
    """
    A mid-slate clock: just after the earliest kickoff when a later one exists
    (so the first wave is locked and the rest is open), else just before the
    only kickoff. "Final"/"In Progress" players count as started regardless.
    """
    zone = classic._game_info_timezone()
    with open(entries_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    kickoffs = sorted({
        kickoff
        for row in rows
        for cell in row
        if " ET" in cell and (kickoff := classic.parse_kickoff(cell, zone)) is not None
    })
    if not kickoffs:
        raise SystemExit(f"No kickoff times in {entries_path}; pass --now.")
    if len(kickoffs) > 1:
        return kickoffs[0] + timedelta(minutes=1)
    return kickoffs[0] - timedelta(minutes=30)


def synthetic_entries(source: str, entries: int, contests: int, folder: str) -> str:
    """
    Writes a copy of `source` whose entry rows are its filled entries cycled to
    `entries` rows over `contests` contests (round robin, so contests
    interleave in file order). The pool to the right is kept as is.
    """
    with open(source, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    width = late_swap.ENTRY_SLOT_START_COL + classic.ROSTER_SIZE
    templates = [
        row[:width] for row in rows[1:]
        if row and row[0].strip().isdigit() and any(c.strip() for c in row[4:width])
    ]
    out = [rows[0]]
    body = rows[1:]
    for i in range(max(entries, len(body))):
        right = body[i][width:] if i < len(body) else []
        if i < entries:
            template = templates[i % len(templates)]
            contest_id = str(900000000 + i % contests)
            left = [str(7000000000 + i), f"Bench Contest {contest_id}", contest_id,
                    template[3], *template[4:width]]
        else:
            left = [""] * width
        out.append(left + right)
    path = os.path.join(folder, f"DKEntries-bench-{entries}x{contests}.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(out)
    return path


# --- Runs ---


@dataclass
class LoopTiming:
    label: str
    n: int
    u: int
    built: int
    total: float
    build: float
    rebuild: float
    solve: float
    other: float

    def row(self) -> str:
        solves = max(self.built, 1)
        return (
            f"| {self.label} | {self.n} | {self.u} | {self.built} | {self.total:.2f} "
            f"| {self.build * 1000:.0f} | {self.rebuild / solves * 1000:.1f} "
            f"| {self.solve / solves * 1000:.1f} | {self.other / solves * 1000:.1f} "
            f"| {self.rebuild / self.total:.0%} | {self.solve / self.total:.0%} |"
        )


LOOP_HEADER = (
    "| Format | n | u | built | total s | build ms | rebuild ms/it | solve ms/it "
    "| other ms/it | rebuild % | solve % |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)


def time_loop(label: str, run, n: int, u: int) -> LoopTiming:
    with instrumented() as log:
        start = time.perf_counter()
        result = run(n, u)
        end = time.perf_counter()
    first = log.first_solve_at or end
    rebuild, solve = sum(log.rebuild), sum(log.solve)
    return LoopTiming(
        label, n, u, len(result.lineups), end - start, first - start,
        rebuild, solve, (end - first) - rebuild - solve,
    )


@dataclass
class SwapTiming:
    label: str
    entries: int
    contests: int
    total: float
    rebuild: float
    solve: float
    solves: int

    def row(self) -> str:
        return (
            f"| {self.label} | {self.entries} | {self.contests} | {self.total:.2f} "
            f"| {self.total / max(self.entries, 1) * 1000:.0f} | {self.solves} "
            f"| {self.rebuild:.2f} | {self.solve:.2f} "
            f"| {self.total - self.rebuild - self.solve:.2f} |"
        )


SWAP_HEADER = (
    "| Entries file | entries | contests | total s | ms/entry | solves "
    "| rebuild s | solve s | other s |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
)


def time_swap(label: str, df, entries_path: str, now: datetime, u: int) -> SwapTiming:
    options = late_swap.LateSwapOptions(min_uniques=u)
    with instrumented() as log:
        start = time.perf_counter()
        result = late_swap.run(df, entries_path, options, now)
        total = time.perf_counter() - start
    return SwapTiming(
        label, len(result.outcomes), len(result.slate.contests), total,
        sum(log.rebuild), sum(log.solve), len(log.solve),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixtures", action="store_true", help="use tests/fixtures/public/")
    parser.add_argument("--quick", action="store_true", help="n = 20 only, one synthetic file")
    parser.add_argument("--classic", help="Classic projections CSV")
    parser.add_argument("--showdown", help="Showdown projections CSV")
    parser.add_argument("--entries", help="Classic DKEntries CSV for late swap")
    parser.add_argument("--now", help='late-swap clock, Eastern "YYYY-MM-DD HH:MM"')
    parser.add_argument("--swap-u", type=int, default=2, help="-u for late swap (default 2)")
    parser.add_argument("--skip", nargs="*", default=[], choices=["classic", "showdown", "swap"])
    args = parser.parse_args()

    if args.fixtures:
        paths = dict(FIXTURE_FILES)
    else:
        paths = {
            "classic": next(iter(newest(classic.PROJECTIONS_GLOB)), None),
            "showdown": next(iter(newest(showdown.PROJECTIONS_GLOB)), None),
            "entries": classic_entries_file(newest(common.DK_ENTRIES_GLOB)),
        }
    for key in paths:
        paths[key] = getattr(args, key) or paths[key]
    for key, path in paths.items():
        if not path or not os.path.isfile(path):
            raise SystemExit(f"No {key} file found; pass --{key} or --fixtures.")

    counts = (20,) if args.quick else LINEUP_COUNTS
    zone = classic._game_info_timezone()
    now = (
        datetime.strptime(args.now, "%Y-%m-%d %H:%M").replace(tzinfo=zone)
        if args.now else default_now(paths["entries"])
    )

    print(f"Python {platform.python_version()} on {platform.platform()}, "
          f"{os.cpu_count()} logical CPUs; pulp {version('pulp')}, "
          f"highspy {version('highspy')}, pandas {version('pandas')}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print()

    classic_df = classic.load_player_data(paths["classic"]).df
    classic_df = classic.attach_kickoffs(classic_df, classic.load_dk_kickoffs(paths["entries"]))
    showdown_df = showdown.load_player_data(paths["showdown"]).df
    print(f"Classic pool {len(classic_df)} players, Showdown pool {len(showdown_df)} players.")

    def run_classic(n: int, u: int):
        return classic.run(classic_df, classic.ClassicOptions(num_lineups=n, min_uniques=u))

    def run_showdown(n: int, u: int):
        return showdown.run(showdown_df, showdown.ShowdownOptions(num_lineups=n, min_uniques=u))

    # Warm-up: imports, highspy load, pandas caches.
    run_classic(1, 1)
    run_showdown(1, 1)

    loops = []
    if "classic" not in args.skip:
        loops.append(("Classic", run_classic))
    if "showdown" not in args.skip:
        loops.append(("Showdown", run_showdown))
    if loops:
        print("\n" + LOOP_HEADER)
        for label, run in loops:
            for n in counts:
                for u in MIN_UNIQUES:
                    print(time_loop(label, run, n, u).row(), flush=True)

    if "swap" not in args.skip:
        swap_df = classic.with_ownership(classic_df, classic.OWNERSHIP_LARGE_FIELD)
        print(f"\nLate swap, -u {args.swap_u}, clock {now:%m/%d/%Y %I:%M%p} ET")
        print(SWAP_HEADER)
        print(time_swap(os.path.basename(paths["entries"]), swap_df, paths["entries"],
                        now, args.swap_u).row(), flush=True)
        with tempfile.TemporaryDirectory(prefix="nfl-bench-") as folder:
            for entries, contests in QUICK_SWAP_SCALES if args.quick else SWAP_SCALES:
                path = synthetic_entries(paths["entries"], entries, contests, folder)
                print(time_swap(f"synthetic {entries}x{contests}", swap_df, path,
                                now, args.swap_u).row(), flush=True)


if __name__ == "__main__":
    sys.exit(main())

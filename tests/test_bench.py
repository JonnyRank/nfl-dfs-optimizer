"""
Smoke test for scripts/bench_perf.py: it reaches into the cores (each module's
build_solver binding, ENTRY_SLOT_START_COL, the kickoff helpers), so a core
refactor that breaks it should fail here rather than at the next benchmark.
"""

import importlib.util
import os

from parity_harness import CLASSIC_FILE, LATE_SWAP_ENTRIES, ROOT

from nfl_dfs_optimizer import classic, late_swap

spec = importlib.util.spec_from_file_location(
    "bench_perf", os.path.join(ROOT, "scripts", "bench_perf.py")
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_bench_times_a_lineup_loop_and_a_late_swap(tmp_path):
    df = classic.load_player_data(CLASSIC_FILE).df
    solver = late_swap.build_solver

    loop = bench.time_loop(
        "Classic", lambda n, u: classic.run(df, classic.ClassicOptions(num_lineups=n, min_uniques=u)),
        2, 1,
    )
    assert loop.built == 2 and loop.solves == 2
    assert loop.solve > 0 and loop.rebuild > 0

    path = bench.synthetic_entries(LATE_SWAP_ENTRIES, 4, 2, str(tmp_path))
    swap_df = classic.with_ownership(df, classic.OWNERSHIP_LARGE_FIELD)
    swap = bench.time_swap("synthetic", swap_df, path, bench.default_now(path), 2)
    assert swap.entries == 4 and swap.contests == 2 and swap.solves > 0
    # instrumented() puts every core's own build_solver back.
    assert late_swap.build_solver is solver

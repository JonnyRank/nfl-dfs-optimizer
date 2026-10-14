"""
Late swap across worker processes: rebuilding contests in parallel must give
exactly what the in-process rebuild gives -- the same outcomes in the same
order, and a byte-identical upload file.

The multi-contest inputs are the late-swap fixture's entries cycled over more
contests (contests interleaved in file order, as DraftKings files can be).
"""

import csv
import os
import sys
import threading
import types
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from parity_harness import CLASSIC_FILE, GOLDEN, LATE_SWAP_ENTRIES

from nfl_dfs_optimizer import classic, gui, late_swap
from nfl_dfs_optimizer.late_swap import LateSwapOptions

AT_1_30 = datetime(2026, 9, 27, 13, 30, tzinfo=ZoneInfo("America/New_York"))


@pytest.fixture(scope="module")
def classic_df():
    return classic.with_ownership(
        classic.load_player_data(CLASSIC_FILE).df, classic.OWNERSHIP_LARGE_FIELD
    )


@pytest.fixture(scope="module")
def multi_contest(tmp_path_factory) -> str:
    """12 entries over 3 interleaved contests, built from the fixture's entries."""
    with open(LATE_SWAP_ENTRIES, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    width = late_swap.ENTRY_SLOT_START_COL + classic.ROSTER_SIZE
    templates = [row[:width] for row in rows[1:] if row and row[0].isdigit() and row[4]]
    body = rows[1:]
    out = [rows[0]]
    for i in range(max(12, len(body))):
        right = body[i][width:] if i < len(body) else []
        if i < 12:
            template = templates[i % len(templates)]
            contest = str(800000000 + i % 3)
            left = [str(6000000000 + i), f"Contest {contest}", contest, *template[3:width]]
        else:
            left = [""] * width
        out.append(left + right)
    path = tmp_path_factory.mktemp("entries") / "DKEntries-multi.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(out)
    return str(path)


def summary(result: late_swap.LateSwapResult) -> list[tuple]:
    return [
        (o.number, o.entry.entry_id, o.original_ids, o.final_ids, o.locked, o.status, o.note)
        for o in result.outcomes
    ]


@pytest.mark.parametrize(
    "options",
    [
        LateSwapOptions(min_uniques=2),
        LateSwapOptions(min_uniques=3, min_salary=49500, stack=1),
        LateSwapOptions(min_uniques=4, no_dst_opp=True, max_te=1),
    ],
    ids=["u2", "u3_floor_stack", "u4_ndo_te"],
)
def test_parallel_matches_in_process(classic_df, multi_contest, options):
    serial = late_swap.run(classic_df, multi_contest, options, AT_1_30, workers=1)
    parallel = late_swap.run(classic_df, multi_contest, options, AT_1_30, workers=3)
    assert summary(parallel) == summary(serial)
    assert late_swap.upload_csv(parallel.slate, parallel.outcomes) == late_swap.upload_csv(
        serial.slate, serial.outcomes
    )
    # The caller's own entry objects come back, not the workers' copies.
    assert all(o.entry is e for o, e in zip(parallel.outcomes, parallel.slate.entries, strict=True))


def test_the_pool_really_runs(classic_df, multi_contest, monkeypatch):
    # The fallback would hide a pool that never starts. Workers import the
    # module fresh, so only an in-process rebuild sees this patch.
    def in_process(*args, **kwargs):
        raise AssertionError("rebuilt in-process, not in the workers")

    monkeypatch.setattr(late_swap, "_rebuild_contest", in_process)
    result = late_swap.run(classic_df, multi_contest, LateSwapOptions(), AT_1_30, workers=3)
    assert len(result.outcomes) == 12


def test_parallel_with_skipped_entries(classic_df, multi_contest):
    swap = late_swap.load_slate(classic_df, multi_contest, AT_1_30)
    chosen = frozenset(entry.entry_id for entry in swap.entries[::2])
    options = LateSwapOptions(min_uniques=3, entry_ids=chosen)
    serial = late_swap.run(classic_df, multi_contest, options, AT_1_30, workers=1)
    parallel = late_swap.run(classic_df, multi_contest, options, AT_1_30, workers=2)
    assert summary(parallel) == summary(serial)
    assert [o.number for o in parallel.outcomes] == list(range(1, len(chosen) + 1))


def test_parallel_matches_the_cli_golden(classic_df):
    options = LateSwapOptions(min_uniques=2, min_salary=49500, stack=1)
    result = late_swap.run(classic_df, LATE_SWAP_ENTRIES, options, AT_1_30, workers=2)
    path = os.path.join(GOLDEN, "classic", "late_swap_1pm.export.csv")
    with open(path, encoding="utf-8") as handle:
        golden = handle.read().split("\n", 1)[1]
    assert late_swap.upload_csv(result.slate, result.outcomes).replace("\r\n", "\n") == golden


def test_a_failed_pool_falls_back_to_in_process(classic_df, multi_contest, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("no processes here")

    monkeypatch.setattr(late_swap, "ProcessPoolExecutor", broken)
    options = LateSwapOptions(min_uniques=2)
    fallback = late_swap.run(classic_df, multi_contest, options, AT_1_30, workers=3)
    serial = late_swap.run(classic_df, multi_contest, options, AT_1_30, workers=1)
    assert summary(fallback) == summary(serial)


def test_parallel_workers(monkeypatch):
    monkeypatch.setattr(late_swap.os, "process_cpu_count", lambda: 4)
    threshold = late_swap.PARALLEL_MIN_ENTRIES
    assert late_swap.parallel_workers(None, contests=5, entries=threshold - 1) == 1
    assert late_swap.parallel_workers(None, contests=1, entries=threshold) == 1
    assert late_swap.parallel_workers(None, contests=3, entries=threshold) == 3
    assert late_swap.parallel_workers(None, contests=10, entries=threshold) == 4  # CPUs and cap
    assert late_swap.parallel_workers(8, contests=10, entries=threshold) == 4  # CPUs
    assert late_swap.parallel_workers(1, contests=10, entries=threshold) == 1
    assert late_swap.parallel_workers(2, contests=10, entries=5) == 2  # explicit skips the threshold
    monkeypatch.setattr(late_swap.os, "process_cpu_count", lambda: None)
    assert late_swap.parallel_workers(None, contests=10, entries=threshold) == 1


def test_workers_never_import_the_main_script(classic_df, multi_contest, tmp_path, monkeypatch):
    # Under Streamlit, __main__ is the app script; a worker importing it would
    # run the whole app. This stand-in leaves a file behind if it is imported.
    marker = tmp_path / "imported.txt"
    script = tmp_path / "fake_app.py"
    script.write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = str(script)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    result = late_swap.run(classic_df, multi_contest, LateSwapOptions(), AT_1_30, workers=2)
    assert len(result.outcomes) == 12
    assert not marker.exists()
    assert sys.modules["__main__"] is fake_main


def test_a_worker_error_reaches_the_caller_without_a_rerun(classic_df, multi_contest, monkeypatch):
    # A candidate with no projection makes the rebuild itself raise KeyError,
    # in a worker. That is a bug to report, not a pool failure to retry.
    slate = late_swap.load_slate(classic_df, multi_contest, AT_1_30)
    candidates = dict(slate.candidates(set()))
    stray = next(p for i, p in slate.pool.items() if i not in slate.projections and not p.started)
    candidates[stray.dk_id] = stray

    def in_process(*args, **kwargs):
        raise AssertionError("retried in-process")

    monkeypatch.setattr(late_swap, "_rebuild_contest", in_process)
    with pytest.raises(KeyError):
        late_swap.rebuild_entries(slate, LateSwapOptions(), candidates, None, workers=2)


def test_an_unpicklable_slate_falls_back_to_in_process(classic_df, multi_contest):
    slate = late_swap.load_slate(classic_df, multi_contest, AT_1_30)
    candidates = slate.candidates(set())
    solver = late_swap.build_solver()
    serial = late_swap.rebuild_entries(slate, LateSwapOptions(), candidates, solver, workers=1)
    slate.lock = threading.Lock()  # what a careless future field might add
    fallback = late_swap.rebuild_entries(slate, LateSwapOptions(), candidates, solver, workers=2)
    assert [(o.number, o.final_ids, o.status) for o in fallback] == [
        (o.number, o.final_ids, o.status) for o in serial
    ]


def test_a_main_replaced_inside_the_window_is_left_alone(monkeypatch):
    original, replacement = types.ModuleType("__main__"), types.ModuleType("__main__")
    monkeypatch.setitem(sys.modules, "__main__", original)
    with late_swap._main_hidden_from_spawn():
        assert sys.modules["__main__"] is not original
        sys.modules["__main__"] = replacement  # e.g. another Streamlit script run
    assert sys.modules["__main__"] is replacement


@pytest.mark.parametrize(("hosted", "workers"), [("true", 1), ("false", None)])
def test_hosted_late_swap_stays_in_process(classic_df, monkeypatch, hosted, workers):
    monkeypatch.setenv("NFL_DFS_HOSTED", hosted)
    seen = []
    real_run = late_swap.run

    def spy(*args, **kwargs):
        seen.append(args[4] if len(args) > 4 else kwargs.get("workers"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(late_swap, "run", spy)
    settings = gui.Settings(min_uniques=2, dk_entries=LATE_SWAP_ENTRIES)
    run = gui.run_late_swap(
        classic.load_player_data(CLASSIC_FILE).df, CLASSIC_FILE, settings, {}, {}, None,
        entry_ids=None, now=AT_1_30,
    )
    assert run.error is None, run.error
    assert seen == [workers]

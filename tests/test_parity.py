"""
Parity tests: every case in parity_harness.CASES must reproduce its goldens.

test_summary is the contract -- lineup count, and per lineup the objective
score and salary. test_transcript and test_export pin the CLI's printed and
exported output byte for byte (after path/time normalization).
"""

import pytest
from parity_harness import CASES, load_golden, run_case, summarize

SCORE_TOLERANCE = 0.01


@pytest.fixture(scope="module")
def outputs():
    """Runs each case once per module; the three tests share its output."""
    cache = {}

    def get(case):
        if case not in cache:
            cache[case] = run_case(case)
        return cache[case]

    return get


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.fmt}/{c.name}")
def test_summary(case, outputs):
    golden, _, _ = load_golden(case)
    actual = summarize(case, outputs(case).transcript)
    assert actual["count"] == golden["count"]
    for number, (got, want) in enumerate(zip(actual["lineups"], golden["lineups"]), start=1):
        assert got["score"] == pytest.approx(want["score"], abs=SCORE_TOLERANCE), number
        assert got["salary"] == want["salary"], number


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c.fmt}/{c.name}")
def test_transcript(case, outputs):
    _, transcript, _ = load_golden(case)
    assert outputs(case).transcript == transcript


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.exports], ids=lambda c: f"{c.fmt}/{c.name}"
)
def test_export(case, outputs):
    _, _, export = load_golden(case)
    assert outputs(case).export == export

"""
Tests for the optimizer cores called directly, without the CLI: they reproduce
the parity goldens, never print, validate their input, and take locks and
excludes as player IDs (Classic) or DataFrame rows (Showdown).
"""

import json
import os

import pandas as pd
import pytest
from parity_harness import CLASSIC_FILE, GOLDEN, SHOWDOWN_FILE

from nfl_dfs_optimizer import classic, showdown
from nfl_dfs_optimizer.common import (
    SALARY_CAP,
    TARGET_BLEND,
    TARGET_CEILING,
    PlayerDataError,
)


@pytest.fixture(scope="module")
def classic_pool():
    return classic.load_player_data(CLASSIC_FILE)


@pytest.fixture(scope="module")
def showdown_pool():
    return showdown.load_player_data(SHOWDOWN_FILE)


def golden(fmt: str, name: str) -> dict:
    with open(os.path.join(GOLDEN, fmt, f"{name}.json"), encoding="utf-8") as handle:
        return json.load(handle)


def assert_matches_golden(result, fmt: str, name: str) -> None:
    want = golden(fmt, name)
    assert len(result.lineups) == want["count"]
    for lineup, expected in zip(result.lineups, want["lineups"]):
        assert lineup.score == pytest.approx(expected["score"], abs=0.01)
        assert lineup.salary == expected["salary"]


def ids_for(df: pd.DataFrame, *names: str) -> tuple[int, ...]:
    return tuple(int(df.loc[df["Player"] == name, "ID"].iloc[0]) for name in names)


def row_for(df: pd.DataFrame, name: str):
    return df.index[df["Player"] == name][0]


# --- Classic ---


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("multi_u2", {"num_lineups": 10, "min_uniques": 2}),
        ("stack2", {"num_lineups": 3, "min_uniques": 2, "stack": 2}),
        ("no_dst_opp", {"num_lineups": 3, "min_uniques": 2, "no_dst_opp": True}),
        ("min_salary", {"num_lineups": 5, "min_uniques": 2, "min_salary": 49900}),
        ("ceiling", {"num_lineups": 3, "min_uniques": 2, "target": TARGET_CEILING}),
        ("projceiling", {"num_lineups": 3, "min_uniques": 2, "target": TARGET_BLEND}),
    ],
)
def test_classic_run_matches_golden(classic_pool, name, kwargs):
    result = classic.run(classic_pool.df, classic.ClassicOptions(**kwargs))
    assert result.status == "Optimal" and result.complete
    assert_matches_golden(result, "classic", name)


def test_classic_locks_and_excludes_by_id(classic_pool):
    df = classic_pool.df
    options = classic.ClassicOptions(
        num_lineups=3,
        min_uniques=2,
        lock_ids=ids_for(df, "Lamar Jackson"),
        exclude_ids=ids_for(df, "Derrick Henry"),
    )
    result = classic.run(df, options)
    assert_matches_golden(result, "classic", "lock_exclude")
    for lineup in result.lineups:
        names = {p["Player"] for p in lineup.players.values()}
        assert "Lamar Jackson" in names and "Derrick Henry" not in names


def test_classic_infeasible_reports_status_and_explanation(classic_pool):
    df = classic_pool.df
    options = classic.ClassicOptions(
        num_lineups=3, lock_ids=ids_for(df, "Josh Allen", "Lamar Jackson")
    )
    result = classic.run(df, options)
    assert result.lineups == [] and not result.complete
    assert result.status == "Infeasible"
    assert "This means no lineup exists that satisfies the base constraints." in result.messages


def test_classic_unknown_id_raises(classic_pool):
    with pytest.raises(ValueError, match="player ID 1"):
        classic.run(classic_pool.df, classic.ClassicOptions(lock_ids=(1,)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_salary": -1}, "cannot be negative"),
        ({"min_salary": SALARY_CAP + 1}, "exceeds the"),
        ({"target": "median"}, "Unknown optimization target"),
        ({"ownership_field": "medium"}, "Unknown ownership field"),
    ],
)
def test_classic_options_validate(kwargs, message):
    with pytest.raises(ValueError, match=message):
        classic.ClassicOptions(**kwargs).validate()


def test_classic_keeps_both_ownership_fields(classic_pool):
    df = classic_pool.df
    raw = pd.read_csv(CLASSIC_FILE, encoding="utf-8-sig")
    gibbs = raw[raw["Player"] == "Jahmyr Gibbs"].iloc[0]
    row = df[df["Player"] == "Jahmyr Gibbs"].iloc[0]
    assert row["OwnershipLarge"] == pytest.approx(float(gibbs["Large Field"].rstrip("%")))
    assert row["OwnershipSmall"] == pytest.approx(float(gibbs["Small Field"].rstrip("%")))

    # The field only changes the displayed ownership, never the lineups.
    large = classic.run(df, classic.ClassicOptions(num_lineups=2))
    small = classic.run(
        df, classic.ClassicOptions(num_lineups=2, ownership_field=classic.OWNERSHIP_SMALL_FIELD)
    )
    assert [lu.score for lu in large.lineups] == [lu.score for lu in small.lineups]
    for lineup in small.lineups:
        for player in lineup.players.values():
            assert player["Ownership"] == player["OwnershipSmall"]


def write_csv(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "projections.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


BASE_ROW = {
    "Player": "A", "Position": "QB", "Team": "BUF", "Opp": "MIA",
    "Salary": "$6,000", "Projection": "20", "ID": 1,
}


def test_small_field_never_falls_back_to_unlabeled_own(tmp_path):
    pool = classic.load_player_data(write_csv(tmp_path, [{**BASE_ROW, "Own": "12.5%"}]))
    row = pool.df.iloc[0]
    assert row["OwnershipLarge"] == 12.5
    assert row["OwnershipSmall"] == 0.0
    assert any("No small field ownership column" in n for n in pool.notes["small"])
    assert not any("ownership column" in n for n in pool.notes["large"])


def test_small_field_supersedes_literal_ownership_column(tmp_path):
    path = write_csv(tmp_path, [{**BASE_ROW, "Ownership": "30%", "Small Field": "5%"}])
    pool = classic.load_player_data(path)
    row = pool.df.iloc[0]
    assert (row["OwnershipLarge"], row["OwnershipSmall"]) == (30.0, 5.0)
    assert any("Ignoring the file's own 'Ownership'" in n for n in pool.notes["small"])
    assert not any("Ignoring" in n for n in pool.notes["large"])


def test_missing_required_column_raises_with_notes(tmp_path):
    row = {k: v for k, v in BASE_ROW.items() if k != "Opp"}
    with pytest.raises(PlayerDataError, match="Opp") as info:
        classic.load_player_data(write_csv(tmp_path, [row]))
    assert info.value.notes["large"][0].startswith("Successfully loaded 1 players")


def test_classic_lineups_df(classic_pool):
    result = classic.run(classic_pool.df, classic.ClassicOptions(num_lineups=2, min_uniques=2))
    frame = result.lineups_df
    assert len(frame) == 18
    assert list(frame["Slot"][:9]) == ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    assert frame.groupby("Lineup")["Salary"].sum().tolist() == [lu.salary for lu in result.lineups]


# --- Showdown ---


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("multi_u2", {"num_lineups": 10, "min_uniques": 2}),
        ("salary_window", {"num_lineups": 5, "min_uniques": 2, "max_salary": 49800, "min_salary": 49500}),
        ("ceiling", {"num_lineups": 5, "min_uniques": 2, "target": TARGET_CEILING}),
        ("projceiling", {"num_lineups": 5, "min_uniques": 3, "target": TARGET_BLEND}),
    ],
)
def test_showdown_run_matches_golden(showdown_pool, name, kwargs):
    result = showdown.run(showdown_pool.df, showdown.ShowdownOptions(**kwargs))
    assert result.complete
    assert_matches_golden(result, "showdown", name)


def test_showdown_slot_locks_and_excludes_by_row(showdown_pool):
    df = showdown_pool.df
    options = showdown.ShowdownOptions(
        num_lineups=4,
        min_uniques=2,
        locks=(showdown.SlotSelection((row_for(df, "Drake London"),), "CPT"),),
        excludes=(showdown.SlotSelection((row_for(df, "Jordan Love"),), "FLEX"),),
    )
    result = showdown.run(df, options)
    assert_matches_golden(result, "showdown", "lock_exclude_slots")
    for lineup in result.lineups:
        assert lineup.rows[0]["Player"] == "Drake London"
        assert "Jordan Love" not in [r["Player"] for r in lineup.rows[1:]]


def test_showdown_unknown_row_raises(showdown_pool):
    options = showdown.ShowdownOptions(locks=(showdown.SlotSelection((9999,), None),))
    with pytest.raises(ValueError, match="9999"):
        showdown.run(showdown_pool.df, options)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_lineups": 0}, "--num-lineups"),
        ({"min_uniques": 7}, "--min-uniques"),
        ({"max_salary": 0}, "--max-salary"),
        ({"min_salary": -5}, "cannot be negative"),
        ({"max_salary": 45000, "min_salary": 46000}, "cannot exceed"),
    ],
)
def test_showdown_options_validate(kwargs, message):
    with pytest.raises(ValueError, match=message):
        showdown.ShowdownOptions(**kwargs).validate()


def test_showdown_max_salary_clamps_with_note():
    notes: list[str] = []
    options = showdown.ShowdownOptions(max_salary=60000)
    options.validate(notes)
    assert options.effective_max_salary == SALARY_CAP
    assert "Clamping to $50,000" in notes[0]


def test_slot_selection_rejects_bad_slot():
    with pytest.raises(ValueError, match="Slot must be"):
        showdown.SlotSelection((0,), "QB")


# --- Silence ---


def test_cores_never_print(capsys, tmp_path):
    pool = classic.load_player_data(CLASSIC_FILE)
    result = classic.run(
        pool.df, classic.ClassicOptions(num_lineups=2, target=TARGET_CEILING, min_salary=49000)
    )
    classic.export_rows(result, None)
    sd_pool = showdown.load_player_data(SHOWDOWN_FILE)
    sd_result = showdown.run(sd_pool.df, showdown.ShowdownOptions(num_lineups=50, min_uniques=6))
    showdown.export_rows(sd_result, None)
    assert not sd_result.complete  # exercised the stop path too
    assert capsys.readouterr() == ("", "")

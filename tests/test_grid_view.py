"""
The player grid's view controls: the FLEX filter and Position order, Sort by,
the Columns picker, Clear filters, and the entries grid's Lineup width.
"""

import pandas as pd
import pytest
from parity_harness import CLASSIC_FILE, SHOWDOWN_FILE
from test_gui import downloads, key_of, run_app  # noqa: F401 -- downloads is a fixture

from nfl_dfs_optimizer import classic, gui, showdown
from nfl_dfs_optimizer.gui import CLASSIC, EXCLUDE, LATE_SWAP, LOCK, SHOWDOWN


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    return classic.load_player_data(CLASSIC_FILE).df


@pytest.fixture(scope="module")
def showdown_players() -> pd.DataFrame:
    return showdown.load_player_data(SHOWDOWN_FILE).df


def grid(app) -> pd.DataFrame:
    """The player grid as drawn (the first table on the page)."""
    value = app.dataframe[0].value
    return value.data if hasattr(value, "data") else value


def test_position_options_order_and_flex():
    assert gui.position_options(CLASSIC, ["WR", "DST", "QB", "TE", "RB"]) == [
        "QB", "RB", "WR", "TE", "FLEX", "DST",
    ]
    assert gui.position_options(SHOWDOWN, ["K", "WR", "DST", "QB", "TE", "RB"]) == [
        "QB", "RB", "WR", "TE", "K", "DST",
    ]
    # FLEX follows whichever of QB/RB/WR/TE are present; unknowns go last.
    assert gui.position_options(CLASSIC, ["WR", "QB", "LS"]) == ["QB", "WR", "FLEX", "LS"]
    assert gui.position_options(CLASSIC, ["QB", "DST"]) == ["QB", "DST"]


def test_flex_filter_is_rb_wr_te(players):
    frame = gui.grid_frame(CLASSIC, players, {}, {})
    flex = gui.filter_frame(frame, ["FLEX"], [], "")
    assert set(flex["Position"]) == {"RB", "WR", "TE"}
    assert len(flex) == frame["Position"].isin(["RB", "WR", "TE"]).sum()
    with_qb = gui.filter_frame(frame, ["FLEX", "QB"], [], "")
    assert set(with_qb["Position"]) == {"QB", "RB", "WR", "TE"}


def test_sort_frame(players):
    frame = gui.grid_frame(CLASSIC, players, {}, {})
    down = gui.sort_frame(frame, "Salary", True)
    assert down["Salary"].is_monotonic_decreasing
    assert gui.sort_frame(frame, "Salary", False)["Salary"].is_monotonic_increasing
    # Ties keep file order.
    for _, group in down.groupby("Salary"):
        assert list(group.index) == sorted(group.index, key=list(frame.index).index)
    names = gui.sort_frame(frame, "Player", False)["Player"].str.lower()
    assert names.is_monotonic_increasing
    assert gui.sort_frame(frame, "Nope", True) is frame
    assert LOCK not in gui.sort_columns(frame.columns) and gui.EDITED not in gui.sort_columns(frame.columns)


def test_grid_key_changes_with_sort():
    base = gui.grid_key(0, CLASSIC, "a.csv", 1.0, [], [], "", ("Salary", True))
    assert base == gui.grid_key(0, CLASSIC, "a.csv", 1.0, [], [], "", ("Salary", True))
    assert base != gui.grid_key(0, CLASSIC, "a.csv", 1.0, [], [], "", ("Salary", False))
    assert base != gui.grid_key(0, CLASSIC, "a.csv", 1.0, [], [], "", ("Projection", True))


def test_edits_on_a_sorted_grid_map_to_the_right_player(players):
    frame = gui.grid_frame(CLASSIC, players, {}, {})
    shown = gui.sort_frame(frame, "Salary", True)
    locks, excludes = {}, {}
    gui.apply_grid_edits(CLASSIC, shown, {0: {LOCK: True}}, players, locks, excludes, {})
    top = shown.iloc[0]
    assert set(locks) == {key_of(CLASSIC, players, top["Player"])}


def test_grid_columns_match_the_grid(players, showdown_players):
    assert gui.grid_columns(CLASSIC) == list(gui.grid_frame(CLASSIC, players, {}, {}).columns)
    showdown_grid = gui.grid_frame(SHOWDOWN, showdown_players, {}, {})
    assert gui.grid_columns(SHOWDOWN) == list(showdown_grid.columns)


def test_column_order(players):
    frame = gui.grid_frame(CLASSIC, players, {}, {})
    hideable = gui.hideable_columns(frame.columns)
    assert LOCK not in hideable and EXCLUDE not in hideable and "Player" not in hideable
    assert gui.column_order(frame, hideable) == list(frame.columns)
    assert gui.column_order(frame, ["Projection", "Salary"]) == [
        LOCK, EXCLUDE, "Player", "Salary", "Projection",
    ]


def test_lineup_column_width():
    short = pd.DataFrame({"Lineup": ["a, b"]})
    assert gui.lineup_column_width(short) == gui.LINEUP_MIN_WIDTH
    long = pd.DataFrame({"Lineup": ["x" * 200, "y"]})
    assert gui.lineup_column_width(long) == round(200 * gui.LINEUP_CHAR_WIDTH) + gui.LINEUP_PADDING
    assert gui.lineup_column_width(pd.DataFrame({"Lineup": []})) == gui.LINEUP_MIN_WIDTH


@pytest.mark.usefixtures("downloads")
def test_app_grid_starts_by_salary_descending():
    app = run_app()
    shown = grid(app)
    assert shown["Salary"].is_monotonic_decreasing
    assert app.selectbox(key="w_Classic_sort_by").value == "Salary"
    assert app.multiselect(key="w_Classic_positions").options[:6] == [
        "QB", "RB", "WR", "TE", "FLEX", "DST",
    ]


@pytest.mark.usefixtures("downloads")
def test_app_sort_holds_through_filters():
    app = run_app()
    app.selectbox(key="w_Classic_sort_by").set_value("Projection").run()
    app.selectbox(key="w_Classic_sort_order").set_value("Low to high").run()
    app.multiselect(key="w_Classic_positions").set_value(["FLEX"]).run()
    app.text_input(key="w_Classic_search").input("a").run()
    shown = grid(app)
    assert set(shown["Position"]) <= {"RB", "WR", "TE"}
    assert shown["Projection"].is_monotonic_increasing
    assert shown["Player"].str.lower().str.contains("a").all()


@pytest.mark.usefixtures("downloads")
def test_app_clear_filters():
    app = run_app()
    clear = next(b for b in app.button if b.label == "Clear filters")
    assert clear.disabled
    app.multiselect(key="w_Classic_positions").set_value(["QB"]).run()
    app.multiselect(key="w_Classic_teams").set_value(["BUF"]).run()
    app.text_input(key="w_Classic_search").input("allen").run()
    assert len(grid(app)) == 1
    next(b for b in app.button if b.label == "Clear filters").click().run()
    assert not app.exception, app.exception
    assert app.multiselect(key="w_Classic_positions").value == []
    assert app.multiselect(key="w_Classic_teams").value == []
    assert app.text_input(key="w_Classic_search").value == ""
    assert len(grid(app)) > 1


@pytest.mark.usefixtures("downloads")
def test_app_columns_picker_hides_columns():
    app = run_app()
    picker = app.multiselect(key="w_Classic_columns")
    assert picker.value == picker.options  # every column shown by default
    assert picker.options[:2] == ["Position", "Team"] and gui.EDITED in picker.options
    picker.set_value(["Salary", "Projection"]).run()
    assert not app.exception, app.exception
    assert app.multiselect(key="w_Classic_columns").value == ["Salary", "Projection"]


@pytest.mark.usefixtures("downloads")
def test_app_showdown_has_no_flex_option():
    app = run_app(w_format=SHOWDOWN)
    options = app.multiselect(key="w_Showdown_positions").options
    assert "FLEX" not in options
    assert grid(app)["Salary"].is_monotonic_decreasing


@pytest.mark.usefixtures("downloads")
def test_app_late_swap_has_flex_option():
    app = run_app(w_format=LATE_SWAP)
    assert "FLEX" in app.multiselect(key="w_Late swap_positions").options


def test_editing_the_sorted_column_moves_rows(players):
    """
    An edit to the sorted column re-sorts the grid, so the app must start a
    fresh editor: folding the same pending edit again would hit another player.
    """
    shown = gui.sort_frame(gui.grid_frame(CLASSIC, players, {}, {}), "Projection", True)
    edits = {}
    pending = {0: {"Projection": 0.5}}  # the top projection drops to the bottom
    outcome = gui.apply_grid_edits(CLASSIC, shown, pending, players, {}, {}, edits)
    assert outcome.changed
    resorted = gui.sort_frame(gui.grid_frame(CLASSIC, players, {}, {}, edits), "Projection", True)
    assert gui.rows_moved(shown, resorted)
    # Without the fresh editor, the same pending edit lands on the new top player.
    gui.apply_grid_edits(CLASSIC, resorted, pending, players, {}, {}, edits)
    assert len(edits) == 2


def test_editing_another_column_keeps_rows(players):
    shown = gui.sort_frame(gui.grid_frame(CLASSIC, players, {}, {}), "Salary", True)
    edits = {}
    gui.apply_grid_edits(CLASSIC, shown, {0: {"Projection": 0.5}}, players, {}, {}, edits)
    resorted = gui.sort_frame(gui.grid_frame(CLASSIC, players, {}, {}, edits), "Salary", True)
    assert not gui.rows_moved(shown, resorted)

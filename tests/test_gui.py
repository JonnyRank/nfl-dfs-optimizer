"""
Tests for the Streamlit app: gui.py's logic directly, and app.py run headless
through Streamlit's AppTest against a Downloads folder holding the public
fixtures.
"""

import json
import os
import shutil

import pandas as pd
import pytest
from parity_harness import (
    CLASSIC_ENTRIES,
    CLASSIC_FILE,
    GOLDEN,
    SHOWDOWN_ENTRIES,
    SHOWDOWN_FILE,
)
from streamlit.testing.v1 import AppTest

from nfl_dfs_optimizer import classic, common, gui, showdown
from nfl_dfs_optimizer.gui import CLASSIC, EXCLUDE, LOCK, SHOWDOWN

APP = os.path.join(os.path.dirname(gui.__file__), "app.py")


def golden(fmt: str, name: str) -> dict:
    with open(os.path.join(GOLDEN, fmt, f"{name}.json"), encoding="utf-8") as handle:
        return json.load(handle)


def scores(result) -> list[tuple[float, int]]:
    return [(round(lu.score, 2), lu.salary) for lu in result.lineups]


def golden_scores(fmt: str, name: str) -> list[tuple[float, int]]:
    return [(lu["score"], lu["salary"]) for lu in golden(fmt, name)["lineups"]]


@pytest.fixture(scope="module")
def classic_df():
    return classic.load_player_data(CLASSIC_FILE).df


@pytest.fixture(scope="module")
def showdown_df():
    return showdown.load_player_data(SHOWDOWN_FILE).df


def key_of(fmt, df, name):
    row = df.index[df["Player"].str.strip() == name][0]
    return gui.player_key(fmt, df.loc[row])


# --- Grid and selections ---


def test_classic_grid_columns(classic_df):
    frame = gui.grid_frame(CLASSIC, classic_df, {}, {})
    assert list(frame.columns) == [
        LOCK, EXCLUDE, "Player", "Position", "Team", "Opp", "Salary", "Projection",
        "Ceiling", "Small Field Own", "Large Field Own", gui.EDITED,
    ]
    assert not frame[LOCK].any()


def test_showdown_grid_columns_and_opp(showdown_df):
    key = key_of(SHOWDOWN, showdown_df, "Jordan Love")
    frame = gui.grid_frame(SHOWDOWN, showdown_df, {key: "CPT"}, {})
    assert {"CPT Salary", "CPT Projection", "CPT Ceiling", "Own", "CPT Own"} <= set(frame.columns)
    love = frame[frame["Player"] == "Jordan Love"].iloc[0]
    assert (love[LOCK], love["Team"], love["Opp"]) == ("CPT", "GB", "ATL")
    assert frame[EXCLUDE].isna().all()


def test_filter_frame(classic_df):
    frame = gui.grid_frame(CLASSIC, classic_df, {}, {})
    shown = gui.filter_frame(frame, ["QB", "WR"], ["BUF"], "all")
    assert set(shown["Player"]) == {"Josh Allen"}
    assert len(gui.filter_frame(frame, [], [], "  ")) == len(frame)


@pytest.mark.parametrize(
    ("lock", "exclude", "conflict"),
    [
        ("Any", "Any", True),
        ("CPT", "CPT", True),
        ("FLEX", "Any", True),
        ("Any", "CPT", False),
        ("CPT", "FLEX", False),
        (None, "Any", False),
    ],
)
def test_conflicts(lock, exclude, conflict):
    assert gui.conflicts(lock, exclude) is conflict


def test_classic_edits_lock_then_exclude_resolves_conflict(classic_df):
    shown = gui.grid_frame(CLASSIC, classic_df, {}, {})
    allen = key_of(CLASSIC, classic_df, "Josh Allen")
    position = list(shown.index).index(shown.index[shown["Player"] == "Josh Allen"][0])
    locks, excludes = {}, {}

    outcome = gui.apply_grid_edits(CLASSIC, shown, {position: {LOCK: True}}, classic_df, locks, excludes)
    assert outcome.changed and not outcome.reset_grid and locks == {allen: "Any"}

    # Re-applying the same (still pending) edit changes nothing.
    again = gui.apply_grid_edits(CLASSIC, shown, {position: {LOCK: True}}, classic_df, locks, excludes)
    assert not again.changed

    outcome = gui.apply_grid_edits(
        CLASSIC, shown, {position: {LOCK: True, EXCLUDE: True}}, classic_df, locks, excludes
    )
    assert excludes == {allen: "Any"} and locks == {}
    assert outcome.reset_grid and "removed the lock" in outcome.notices[0]


def test_showdown_edits_allow_compatible_slots(showdown_df):
    shown = gui.grid_frame(SHOWDOWN, showdown_df, {}, {})
    love = key_of(SHOWDOWN, showdown_df, "Jordan Love")
    position = list(shown["Player"]).index("Jordan Love")
    locks, excludes = {}, {}
    gui.apply_grid_edits(
        SHOWDOWN, shown, {position: {LOCK: "Any", EXCLUDE: "CPT"}}, showdown_df, locks, excludes
    )
    assert (locks, excludes) == ({love: "Any"}, {love: "CPT"})
    outcome = gui.apply_grid_edits(
        SHOWDOWN, shown, {position: {LOCK: "CPT", EXCLUDE: "CPT"}}, showdown_df, locks, excludes
    )
    assert outcome.reset_grid and locks == {love: "CPT"} and excludes == {}
    gui.apply_grid_edits(SHOWDOWN, shown, {position: {LOCK: None}}, showdown_df, locks, excludes)
    assert locks == {}


def test_describe_and_missing_selections(showdown_df):
    love = key_of(SHOWDOWN, showdown_df, "Jordan Love")
    selections = {love: "CPT", "nobody|xyz": "Any"}
    assert gui.describe_selections(SHOWDOWN, showdown_df, selections) == ["Jordan Love (CPT)"]
    assert gui.missing_selections(SHOWDOWN, showdown_df, selections) == 1


# --- Options, validation, running ---


def test_validation_errors_match_cli_rules(tmp_path):
    assert gui.validation_errors(CLASSIC, gui.Settings()) == []
    assert "--min-uniques" in gui.validation_errors(CLASSIC, gui.Settings(min_uniques=10))[0]
    errors = gui.validation_errors(SHOWDOWN, gui.Settings(max_salary=45000, min_salary=46000))
    assert "cannot exceed" in errors[0]
    missing = gui.validation_errors(CLASSIC, gui.Settings(dk_entries=str(tmp_path / "nope.csv")))
    assert "not found" in missing[0]


def test_classic_optimize_matches_cli(classic_df):
    locks = {key_of(CLASSIC, classic_df, "Lamar Jackson"): "Any"}
    excludes = {key_of(CLASSIC, classic_df, "Derrick Henry"): "Any"}
    settings = gui.Settings(num_lineups=3, min_uniques=2, dk_entries=CLASSIC_ENTRIES)
    run = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, settings, locks, excludes)
    assert run.error is None and run.slate == "Main"
    assert scores(run.result) == golden_scores("classic", "lock_exclude")
    assert run.result.show_kickoff
    assert "Kickoffs matched for 362 of 362 projected players." in run.notes


def test_showdown_optimize_matches_cli(showdown_df):
    locks = {key_of(SHOWDOWN, showdown_df, "Drake London"): "CPT"}
    excludes = {key_of(SHOWDOWN, showdown_df, "Jordan Love"): "FLEX"}
    settings = gui.Settings(num_lineups=4, min_uniques=2)
    run = gui.optimize(SHOWDOWN, showdown_df, SHOWDOWN_FILE, settings, locks, excludes)
    assert scores(run.result) == golden_scores("showdown", "lock_exclude_slots")


def test_optimize_reports_infeasible_without_raising(classic_df):
    locks = {key_of(CLASSIC, classic_df, name): "Any" for name in ("Josh Allen", "Lamar Jackson")}
    run = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(num_lineups=2), locks, {})
    assert run.error is None and run.result.lineups == []
    assert run.result.status == "Infeasible" and run.result.messages


def test_optimize_turns_errors_into_messages(classic_df):
    run = gui.optimize(
        CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(target=common.TARGET_CEILING,
        num_lineups=0), {}, {}
    )
    assert run.result is None and "--num-lineups" in run.error


def test_export_writes_the_cli_file(classic_df, showdown_df, tmp_path, monkeypatch):
    monkeypatch.setattr(common, "EXPORT_DIR", str(tmp_path))
    settings = gui.Settings(num_lineups=2, export=True, dk_entries=CLASSIC_ENTRIES)
    run = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, settings, {}, {})
    assert os.path.basename(run.export_path).startswith("nfl_classic_multi_lineups_")
    exported = pd.read_csv(run.export_path)
    assert len(exported) == 2 * 11  # 9 players, TOTAL, upload row

    sd_settings = gui.Settings(num_lineups=2, export=True, target=common.TARGET_BLEND)
    sd_run = gui.optimize(SHOWDOWN, showdown_df, SHOWDOWN_FILE, sd_settings, {}, {})
    assert os.path.basename(sd_run.export_path).startswith("nfl_showdown_multi_lineups_projceiling_")
    assert any("upload rows" in note for note in sd_run.notes)


def test_lineup_table_slot_order(classic_df):
    settings = gui.Settings(dk_entries=CLASSIC_ENTRIES)
    run = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, settings, {}, {})
    table = gui.lineup_table(CLASSIC, run.result.lineups[0], show_kickoff=True)
    assert list(table["Slot"]) == ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    assert "Kickoff (ET)" in table.columns
    totals = gui.lineup_totals(CLASSIC, run.result.lineups[0], common.TARGET_PROJECTION)
    assert totals.startswith("Projection 151.60")


# --- The app, headless ---


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    """A Downloads folder holding the public fixtures; exports go to tmp too."""
    folder = tmp_path / "Downloads"
    folder.mkdir()
    for path in (CLASSIC_FILE, SHOWDOWN_FILE, CLASSIC_ENTRIES, SHOWDOWN_ENTRIES):
        shutil.copy(path, folder)
    # The Classic entries file must be the newest DKEntries match.
    os.utime(folder / os.path.basename(CLASSIC_ENTRIES), (2e9, 2e9))
    monkeypatch.setattr(common, "DOWNLOADS_DIR", str(folder))
    monkeypatch.setattr(common, "EXPORT_DIR", str(tmp_path / "exports"))
    return folder


def run_app(**session) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    for key, value in session.items():
        app.session_state[key] = value
    app.run()
    assert not app.exception, app.exception
    return app


def click_optimize(app: AppTest) -> AppTest:
    next(b for b in app.button if b.label == "Optimize").click().run()
    assert not app.exception, app.exception
    return app


def test_app_classic_optimize(downloads):
    app = run_app()
    assert "DraftKings NFL DFS Projections -- Main Slate.csv" in app.caption[0].value
    app.number_input(key="w_Classic_num_lineups").set_value(3).run()
    app.number_input(key="w_Classic_min_uniques").set_value(2).run()
    click_optimize(app)
    run = app.session_state["results"][CLASSIC]
    assert scores(run.result) == golden_scores("classic", "multi_u2")[:3]
    assert len(app.dataframe) == 1 + 3  # the player grid, then one table per lineup
    assert any("Main Slate Lineup #3" in m.value for m in app.markdown)


def test_app_showdown_locks_and_export(downloads, tmp_path):
    df = showdown.load_player_data(SHOWDOWN_FILE).df
    love = key_of(SHOWDOWN, df, "Jordan Love")
    selections = {CLASSIC: {"locks": {}, "excludes": {}},
                  SHOWDOWN: {"locks": {love: "CPT"}, "excludes": {}}}
    app = run_app(selections=selections, w_format=SHOWDOWN)
    assert any("Jordan Love (CPT)" in m.value for m in app.markdown)
    app.number_input(key="w_Showdown_num_lineups").set_value(5).run()
    app.number_input(key="w_Showdown_min_uniques").set_value(2).run()
    app.toggle(key="w_Showdown_export").set_value(True).run()
    click_optimize(app)
    run = app.session_state["results"][SHOWDOWN]
    assert scores(run.result) == golden_scores("showdown", "lock_cpt")
    assert all(lu.rows[0]["Player"] == "Jordan Love" for lu in run.result.lineups)
    assert os.path.dirname(run.export_path) == str(tmp_path / "exports")


def test_app_shows_validation_errors_and_blocks_optimize(downloads):
    app = run_app(w_format=SHOWDOWN)
    app.number_input(key="w_Showdown_max_salary").set_value(45000).run()
    app.number_input(key="w_Showdown_min_salary").set_value(46000).run()
    assert any("cannot exceed" in e.value for e in app.sidebar.error)
    assert next(b for b in app.button if b.label == "Optimize").disabled


def test_app_settings_survive_a_format_switch(downloads):
    app = run_app()
    app.number_input(key="w_Classic_num_lineups").set_value(7).run()
    app.radio(key="w_format").set_value(SHOWDOWN).run()
    app.radio(key="w_format").set_value(CLASSIC).run()
    assert app.number_input(key="w_Classic_num_lineups").value == 7


def test_app_without_projections_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DOWNLOADS_DIR", str(tmp_path))
    app = run_app()
    assert "No DraftKings NFL DFS Projections*.csv file" in app.info[0].value


# --- Review follow-ups ---


def test_edits_on_a_filtered_grid_map_to_the_right_player(classic_df):
    shown = gui.filter_frame(gui.grid_frame(CLASSIC, classic_df, {}, {}), ["QB"], ["BAL"], "")
    assert list(shown["Player"]) == ["Lamar Jackson", *shown["Player"][1:]]
    locks = {}
    gui.apply_grid_edits(CLASSIC, shown, {0: {LOCK: True}}, classic_df, locks, {})
    assert locks == {key_of(CLASSIC, classic_df, "Lamar Jackson"): "Any"}


def test_showdown_duplicate_rows_share_one_selection(showdown_df):
    doubled = pd.concat([showdown_df, showdown_df[showdown_df["Player"] == "Jordan Love"]])
    doubled.index = range(len(doubled))
    love = key_of(SHOWDOWN, showdown_df, "Jordan Love")
    options = gui.build_options(SHOWDOWN, gui.Settings(), doubled, {love: "CPT"}, {})
    (selection,) = options.locks
    assert len(selection.rows) == 2 and selection.slot == "CPT"


def test_grid_key_changes_with_file_contents_and_filters():
    base = gui.grid_key(0, CLASSIC, "a.csv", 1.0, [], [], "")
    assert base == gui.grid_key(0, CLASSIC, "a.csv", 1.0, [], [], "")
    assert base != gui.grid_key(0, CLASSIC, "a.csv", 2.0, [], [], "")  # re-saved
    assert base != gui.grid_key(0, CLASSIC, "a.csv", 1.0, ["QB"], [], "")
    assert base != gui.grid_key(1, CLASSIC, "a.csv", 1.0, [], [], "")


def test_showdown_ignores_a_bad_entries_path_unless_exporting(tmp_path):
    missing = str(tmp_path / "nope.csv")
    assert gui.validation_errors(SHOWDOWN, gui.Settings(dk_entries=missing)) == []
    assert gui.validation_errors(SHOWDOWN, gui.Settings(dk_entries=missing, export=True))


def test_no_entries_file_selected_note(classic_df):
    run = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(), {}, {})
    assert run.notes == ["No DraftKings entries file selected; the FLEX is not reordered by kickoff."]
    assert not run.result.show_kickoff


def test_browse_fills_the_path_box(downloads, monkeypatch):
    picked = str(downloads / os.path.basename(CLASSIC_FILE))
    calls = []

    def fake_dialog(title, start_in=None):
        calls.append((title, start_in))
        return picked

    monkeypatch.setattr(gui, "browse_for_csv", fake_dialog)
    app = run_app(w_Classic_file="__other__")
    next(b for b in app.button if b.label == "Browse...").click().run()
    assert not app.exception
    assert app.text_input(key="w_Classic_file_other").value == picked
    assert calls == [("Choose the projections file", None)]
    assert any(os.path.basename(picked) in c.value for c in app.sidebar.caption)


def test_browse_cancel_and_failure_keep_the_path(downloads, monkeypatch):
    monkeypatch.setattr(gui, "browse_for_csv", lambda title, start_in=None: None)
    app = run_app(w_Classic_file="__other__", w_Classic_file_other=CLASSIC_FILE)
    next(b for b in app.button if b.label == "Browse...").click().run()
    assert app.text_input(key="w_Classic_file_other").value == CLASSIC_FILE

    def broken(title, start_in=None):
        raise RuntimeError("Couldn't open the file dialog: no display")

    monkeypatch.setattr(gui, "browse_for_csv", broken)
    next(b for b in app.button if b.label == "Browse...").click().run()
    assert not app.exception
    assert any("no display" in w.value for w in app.warning)
    assert app.text_input(key="w_Classic_file_other").value == CLASSIC_FILE


def test_browse_turns_dialog_errors_into_runtime_errors(monkeypatch):
    import tkinter
    from tkinter import filedialog

    class FakeRoot:
        destroyed = False

        def withdraw(self):
            pass

        def attributes(self, *args):
            pass

        def destroy(self):
            FakeRoot.destroyed = True

    def failing_dialog(**kwargs):
        raise tkinter.TclError("dialog failed")

    monkeypatch.setattr(tkinter, "Tk", FakeRoot)
    monkeypatch.setattr(filedialog, "askopenfilename", failing_dialog)
    with pytest.raises(RuntimeError, match="dialog failed"):
        gui.browse_for_csv("Choose")
    assert FakeRoot.destroyed


def test_app_reports_a_deleted_projections_file(downloads, tmp_path):
    other = tmp_path / "gone.csv"
    shutil.copy(CLASSIC_FILE, other)
    app = run_app(w_Classic_file="__other__", w_Classic_file_other=str(other))
    assert not app.error
    other.unlink()
    app.run()
    assert not app.exception
    assert any("Not found" in c.value for c in app.sidebar.caption)
    assert any("Couldn't load" in e.value for e in app.error)

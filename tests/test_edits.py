"""
Phase 3: editable projections. Edits live in memory, lay over a copy of the
loaded frame, never touch the file, and drive the optimizer exactly as if the
file itself had carried the edited numbers.
"""

import hashlib
import os
import shutil

import pandas as pd
import pytest
from parity_harness import CLASSIC_ENTRIES, CLASSIC_FILE, SHOWDOWN_FILE
from streamlit.testing.v1 import AppTest

from nfl_dfs_optimizer import classic, common, gui, showdown
from nfl_dfs_optimizer.common import TARGET_BLEND, TARGET_CEILING
from nfl_dfs_optimizer.gui import CLASSIC, SHOWDOWN

APP = os.path.join(os.path.dirname(gui.__file__), "app.py")


@pytest.fixture(scope="module")
def classic_df():
    return classic.load_player_data(CLASSIC_FILE).df


@pytest.fixture(scope="module")
def showdown_df():
    return showdown.load_player_data(SHOWDOWN_FILE).df


def key_of(fmt, df, name):
    row = df.index[df["Player"].str.strip() == name][0]
    return gui.player_key(fmt, df.loc[row])


def row_of(df, name):
    return df.index[df["Player"].str.strip() == name][0]


def names(fmt, lineup):
    if fmt == SHOWDOWN:
        return [r["Player"] for r in lineup.rows]
    return [str(p["Player"]).strip() for p in lineup.players.values()]


def summary(result):
    return [(round(lu.score, 2), lu.salary) for lu in result.lineups]


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def rewrite_csv(src, dst, player, updates):
    """A copy of a projections CSV with one player's cells replaced."""
    raw = pd.read_csv(src, dtype=str, encoding="utf-8-sig")
    mask = raw["Player"].str.strip() == player
    assert mask.sum() == 1
    for column, value in updates.items():
        raw.loc[mask, column] = value
    raw.to_csv(dst, index=False, encoding="utf-8-sig")
    return str(dst)


# --- Overrides ---


def test_overrides_never_touch_the_loaded_frame_or_the_file(classic_df):
    before_file = sha(CLASSIC_FILE)
    before_frame = classic_df.copy()
    edits = {key_of(CLASSIC, classic_df, "Tyler Shough"): {"Projection": 40.0}}
    edited = gui.apply_overrides(CLASSIC, classic_df, edits)
    gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(num_lineups=2), {}, {}, edits)
    assert edited.at[row_of(classic_df, "Tyler Shough"), "Projection"] == 40.0
    pd.testing.assert_frame_equal(classic_df, before_frame)
    assert sha(CLASSIC_FILE) == before_file


def test_classic_ownership_edits_feed_both_fields(classic_df):
    key = key_of(CLASSIC, classic_df, "Josh Allen")
    edits = {key: {"Small Field Own": 50.0, "Large Field Own": 40.0}}
    row = gui.apply_overrides(CLASSIC, classic_df, edits).loc[row_of(classic_df, "Josh Allen")]
    assert (row["OwnershipSmall"], row["OwnershipLarge"], row["Ownership"]) == (50.0, 40.0, 40.0)


def test_showdown_flex_edit_recalculates_captain_at_1_5x(showdown_df):
    love = row_of(showdown_df, "Jordan Love")
    file_cpt = showdown_df.at[love, "CptProjection"]
    assert file_cpt != showdown_df.at[love, "Projection"] * 1.5  # the file's own CPT value
    edits = {key_of(SHOWDOWN, showdown_df, "Jordan Love"): {"Projection": 20.0, "Ceiling": 30.0}}
    row = gui.apply_overrides(SHOWDOWN, showdown_df, edits).loc[love]
    assert (row["CptProjection"], row["CptCeiling"]) == (30.0, 45.0)
    # Nobody else's Captain value moves.
    other = row_of(showdown_df, "Bijan Robinson")
    edited = gui.apply_overrides(SHOWDOWN, showdown_df, edits)
    assert edited.at[other, "CptProjection"] == showdown_df.at[other, "CptProjection"]


def test_showdown_ownership_edit_recomputes_flex_ownership(showdown_df):
    edits = {key_of(SHOWDOWN, showdown_df, "Jordan Love"): {"Own": 70.0, "CPT Own": 20.0}}
    row = gui.apply_overrides(SHOWDOWN, showdown_df, edits).loc[row_of(showdown_df, "Jordan Love")]
    assert row["FlexOwnership"] == pytest.approx(50.0)


# --- Grid edits ---


def position_of(shown, name):
    return list(shown["Player"]).index(name)


def test_value_edits_fold_into_edits(classic_df):
    shown = gui.grid_frame(CLASSIC, classic_df, {}, {})
    pos = position_of(shown, "Tyler Shough")
    key = key_of(CLASSIC, classic_df, "Tyler Shough")
    edits: gui.Edits = {}

    outcome = gui.apply_grid_edits(
        CLASSIC, shown, {pos: {"Projection": 25.5}}, classic_df, {}, {}, edits
    )
    assert outcome.changed and edits == {key: {"Projection": 25.5}}
    # The pending value, seen again on the next run, changes nothing.
    again = gui.apply_grid_edits(CLASSIC, shown, {pos: {"Projection": 25.5}}, classic_df, {}, {}, edits)
    assert not again.changed

    # Typing the file's own number back, or clearing the cell, removes the edit.
    file_value = float(classic_df.at[row_of(classic_df, "Tyler Shough"), "Projection"])
    gui.apply_grid_edits(CLASSIC, shown, {pos: {"Projection": file_value}}, classic_df, {}, {}, edits)
    assert edits == {}
    gui.apply_grid_edits(CLASSIC, shown, {pos: {"Ceiling": 50.0}}, classic_df, {}, {}, edits)
    cleared = gui.apply_grid_edits(CLASSIC, shown, {pos: {"Ceiling": None}}, classic_df, {}, {}, edits)
    assert edits == {} and cleared.reset_grid  # redrawn, so the cell shows the file's value

    # Within display precision of the file's number counts as the file's number.
    gui.apply_grid_edits(
        CLASSIC, shown, {pos: {"Projection": file_value + 0.004}}, classic_df, {}, {}, edits
    )
    assert edits == {}


def test_grid_shows_edits_and_marks_them(showdown_df):
    key = key_of(SHOWDOWN, showdown_df, "Jordan Love")
    frame = gui.grid_frame(SHOWDOWN, showdown_df, {}, {}, {key: {"Projection": 20.0}})
    love = frame[frame["Player"] == "Jordan Love"].iloc[0]
    assert (love["Projection"], love["CPT Projection"], love[gui.EDITED]) == (20.0, 30.0, "Projection")
    styles = gui.grid_styles(SHOWDOWN, frame)
    row = frame.index[frame["Player"] == "Jordan Love"][0]
    assert styles.at[row, "Player"] and styles.at[row, "CPT Projection"]
    assert not styles.at[row, "CPT Ceiling"]
    assert not styles.drop(index=row).to_numpy().any()


def test_showdown_edit_notice_says_captain_recalculated(showdown_df):
    shown = gui.grid_frame(SHOWDOWN, showdown_df, {}, {})
    edits: gui.Edits = {}
    outcome = gui.apply_grid_edits(
        SHOWDOWN, shown, {position_of(shown, "Jordan Love"): {"Projection": 20.0}},
        showdown_df, {}, {}, edits,
    )
    assert outcome.notices == ["Jordan Love: Captain projection recalculated at 1.5x = 30.00."]


def test_showdown_clearing_a_flex_edit_says_captain_is_back(showdown_df):
    shown = gui.grid_frame(SHOWDOWN, showdown_df, {}, {})
    key = key_of(SHOWDOWN, showdown_df, "Jordan Love")
    edits: gui.Edits = {key: {"Ceiling": 40.0}}
    outcome = gui.apply_grid_edits(
        SHOWDOWN, shown, {position_of(shown, "Jordan Love"): {"Ceiling": None}},
        showdown_df, {}, {}, edits,
    )
    assert edits == {} and outcome.reset_grid
    assert outcome.notices == [
        "Jordan Love: ceiling is back to the file's value, and so is its Captain ceiling."
    ]
    love = row_of(showdown_df, "Jordan Love")
    restored = gui.apply_overrides(SHOWDOWN, showdown_df, edits)
    assert restored.at[love, "CptCeiling"] == showdown_df.at[love, "CptCeiling"]


def test_showdown_describe_edits_units(showdown_df):
    key = key_of(SHOWDOWN, showdown_df, "Jordan Love")
    (line,) = gui.describe_edits(SHOWDOWN, showdown_df, {key: {"CPT Own": 5.0, "Ceiling": 30.0}})
    assert line == "Jordan Love: CPT Own 11.75% -> 5.00%; Ceiling 27.96 -> 30.00"


def test_prune_drops_edits_the_file_caught_up_with(classic_df):
    shough = key_of(CLASSIC, classic_df, "Tyler Shough")
    file_value = float(classic_df.at[row_of(classic_df, "Tyler Shough"), "Projection"])
    edits = {shough: {"Projection": file_value, "Ceiling": 60.0}, 1: {"Projection": 5.0}}
    assert gui.prune_edits(CLASSIC, classic_df, edits)
    assert edits == {shough: {"Ceiling": 60.0}, 1: {"Projection": 5.0}}  # absent players kept
    assert not gui.prune_edits(CLASSIC, classic_df, edits)


def test_describe_edits(classic_df):
    edits = {key_of(CLASSIC, classic_df, "Tyler Shough"): {"Projection": 25.5, "Large Field Own": 20.0}}
    (line,) = gui.describe_edits(CLASSIC, classic_df, edits)
    assert line.startswith("Tyler Shough: Projection 19.27 -> 25.50; Large Field Own ")
    assert line.endswith("% -> 20.00%")


# --- Edit, then optimize ---


def test_edited_classic_run_equals_an_edited_file(classic_df, tmp_path):
    """The in-memory edit and the same number written into a file agree."""
    edits = {key_of(CLASSIC, classic_df, "Tyler Shough"): {"Projection": 31.0, "Ceiling": 55.0}}
    settings = gui.Settings(num_lineups=5, min_uniques=2, target=TARGET_BLEND)
    in_memory = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, settings, {}, {}, edits)

    edited_file = rewrite_csv(
        CLASSIC_FILE, tmp_path / os.path.basename(CLASSIC_FILE), "Tyler Shough",
        {"DK Proj": "31.0", "DK Ceiling": "55.0"},
    )
    reference = classic.run(
        classic.load_player_data(edited_file).df,
        classic.ClassicOptions(num_lineups=5, min_uniques=2, target=TARGET_BLEND),
    )
    assert summary(in_memory.result) == summary(reference)
    assert all("Tyler Shough" in names(CLASSIC, lu) for lu in in_memory.result.lineups)


def test_edited_showdown_run_equals_an_edited_file(showdown_df, tmp_path):
    edits = {key_of(SHOWDOWN, showdown_df, "Kaleb Johnson"): {"Projection": 19.0}}
    settings = gui.Settings(num_lineups=5, min_uniques=2)
    in_memory = gui.optimize(SHOWDOWN, showdown_df, SHOWDOWN_FILE, settings, {}, {}, edits)

    # The same edit, done by hand in the file: FLEX projection and a 1.5x CPT Proj.
    edited_file = rewrite_csv(
        SHOWDOWN_FILE, tmp_path / "sd.csv", "Kaleb Johnson", {"Proj": "19.0", "CPT Proj": "28.5"}
    )
    reference = showdown.run(
        showdown.load_player_data(edited_file).df,
        showdown.ShowdownOptions(num_lineups=5, min_uniques=2),
    )
    assert summary(in_memory.result) == summary(reference)
    captained = [lu for lu in in_memory.result.lineups if lu.rows[0]["Player"] == "Kaleb Johnson"]
    assert captained and captained[0].rows[0]["Projection"] == pytest.approx(28.5)


def test_zeroing_a_projection_keeps_the_player_out_of_lineups(classic_df):
    edits = {key_of(CLASSIC, classic_df, "Jahmyr Gibbs"): {"Projection": 0.0}}
    run = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(num_lineups=3), {}, {}, edits)
    assert not any("Jahmyr Gibbs" in names(CLASSIC, lu) for lu in run.result.lineups)


def test_ceiling_edit_moves_ceiling_runs_only(classic_df):
    edits = {key_of(CLASSIC, classic_df, "Chris Olave"): {"Ceiling": 80.0}}
    ceiling = gui.optimize(
        CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(target=TARGET_CEILING), {}, {}, edits
    )
    assert "Chris Olave" in names(CLASSIC, ceiling.result.lineups[0])
    base = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(), {}, {}, {})
    projection = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(), {}, {}, edits)
    assert summary(projection.result) == summary(base.result)


def test_ownership_edits_change_display_not_lineups(classic_df):
    base = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(num_lineups=2), {}, {}, {})
    player = str(base.result.lineups[0].players["QB"]["Player"]).strip()
    edits = {key_of(CLASSIC, classic_df, player): {"Small Field Own": 99.0}}
    settings = gui.Settings(num_lineups=2, ownership_field=classic.OWNERSHIP_SMALL_FIELD)
    run = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, settings, {}, {}, edits)
    assert summary(run.result) == summary(base.result)
    assert run.result.lineups[0].players["QB"]["Ownership"] == 99.0


def test_reset_restores_file_results(classic_df):
    base = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(num_lineups=3), {}, {})
    edits = {key_of(CLASSIC, classic_df, "Tyler Shough"): {"Projection": 40.0}}
    gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(num_lineups=3), {}, {}, edits)
    edits.clear()  # what "Reset to file values" does
    again = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, gui.Settings(num_lineups=3), {}, {}, edits)
    assert summary(again.result) == summary(base.result)


def test_export_carries_edited_values(classic_df, tmp_path, monkeypatch):
    monkeypatch.setattr(common, "EXPORT_DIR", str(tmp_path))
    edits = {key_of(CLASSIC, classic_df, "Tyler Shough"): {"Projection": 40.0}}
    settings = gui.Settings(export=True, dk_entries=CLASSIC_ENTRIES)
    run = gui.optimize(CLASSIC, classic_df, CLASSIC_FILE, settings, {}, {}, edits)
    exported = pd.read_csv(run.export_path)
    shough = exported[exported["Player"] == "Tyler Shough"]
    assert shough["Projection"].astype(float).tolist() == [40.0]  # upload rows make it text
    assert run.edited == {key_of(CLASSIC, classic_df, "Tyler Shough")}
    table = gui.lineup_table(CLASSIC, run.result.lineups[0], True, run.edited)
    assert "Tyler Shough ✎" in table["Player"].tolist()


# --- The app ---


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    folder = tmp_path / "Downloads"
    folder.mkdir()
    for path in (CLASSIC_FILE, SHOWDOWN_FILE, CLASSIC_ENTRIES):
        shutil.copy(path, folder)
    monkeypatch.setattr(common, "DOWNLOADS_DIR", str(folder))
    monkeypatch.setattr(common, "EXPORT_DIR", str(tmp_path / "exports"))
    return folder


def test_app_optimizes_with_edits_and_resets(downloads, showdown_df):
    key = key_of(SHOWDOWN, showdown_df, "Kaleb Johnson")
    edits = {CLASSIC: {}, SHOWDOWN: {key: {"Projection": 19.0}}}
    app = AppTest.from_file(APP, default_timeout=120)
    app.session_state["edits"] = edits
    app.session_state["w_format"] = SHOWDOWN
    app.run()
    assert not app.exception
    assert any("Kaleb Johnson: Projection" in m.value for m in app.markdown)
    assert any("1.5x" in c.value for c in app.caption)

    next(b for b in app.button if b.label == "Optimize").click().run()
    run = app.session_state["results"][SHOWDOWN]
    assert run.edited == {key}
    assert any("✎" in c.value for c in app.caption)

    next(b for b in app.button if b.label == "Reset to file values").click().run()
    assert not app.exception
    assert app.session_state["edits"][SHOWDOWN] == {}
    assert any(m.value.startswith("**Edited (0):**") for m in app.markdown)

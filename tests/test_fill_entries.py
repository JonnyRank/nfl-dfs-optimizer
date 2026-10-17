"""
Filling DraftKings entries with generated lineups: the entries core
(entries.py), the gui.py bookkeeping, and the Fill entries panel on the
Classic and Showdown pages (app.py headless).
"""

import csv
import io
import os
import shutil

import pytest
from parity_harness import (
    CLASSIC_FILE,
    LATE_SWAP_ENTRIES,
    SHOWDOWN_ENTRIES,
    SHOWDOWN_FILE,
)
from streamlit.testing.v1 import AppTest

from nfl_dfs_optimizer import classic, common, entries, gui, showdown
from nfl_dfs_optimizer.gui import CLASSIC, SHOWDOWN

APP = os.path.join(os.path.dirname(gui.__file__), "app.py")
LATE_SWAP_IDS = ["5268167451", "5268167452", "5268167453", "5268167454", "5268167455"]


@pytest.fixture(scope="module")
def classic_run():
    df = classic.load_player_data(CLASSIC_FILE).df
    settings = gui.Settings(num_lineups=3, min_uniques=2, dk_entries=LATE_SWAP_ENTRIES)
    return gui.optimize(CLASSIC, df, CLASSIC_FILE, settings, {}, {})


@pytest.fixture(scope="module")
def showdown_run():
    df = showdown.load_player_data(SHOWDOWN_FILE).df
    settings = gui.Settings(num_lineups=2, min_uniques=2, dk_entries=SHOWDOWN_ENTRIES)
    return gui.optimize(SHOWDOWN, df, SHOWDOWN_FILE, settings, {}, {})


def rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


# --- The core ---


def test_read_entries_takes_the_slots_from_the_header():
    found = entries.read_entries(LATE_SWAP_ENTRIES, classic.ROSTER_SIZE, CLASSIC)
    assert found.slot_labels == ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    assert found.header[:4] == ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
    assert [e.entry_id for e in found.entries][:5] == LATE_SWAP_IDS
    showdown_found = entries.read_entries(SHOWDOWN_ENTRIES, showdown.ROSTER_SIZE, SHOWDOWN)
    assert showdown_found.slot_labels == ["CPT"] + ["FLEX"] * 5
    assert [e.cells for e in showdown_found.entries] == [[""] * 6, [""] * 6]


def test_read_entries_rejects_the_other_format(tmp_path):
    with pytest.raises(ValueError, match="has 6 roster slots .* Classic lineups need"):
        entries.read_entries(SHOWDOWN_ENTRIES, classic.ROSTER_SIZE, CLASSIC)
    not_entries = tmp_path / "x.csv"
    not_entries.write_text("Name,ID\n")
    with pytest.raises(ValueError, match="not a DraftKings entries file"):
        entries.read_entries(str(not_entries))
    with pytest.raises(FileNotFoundError):
        entries.read_entries(str(tmp_path / "missing.csv"))


def test_cell_name_strips_the_id_and_locked_tag():
    assert entries.cell_name("Josh Allen (44220217) (LOCKED)") == "Josh Allen"
    assert entries.cell_name("Titans  (44221417)") == "Titans"
    assert entries.cell_name("") == ""


def test_upload_csv_writes_only_the_given_entries_in_file_order():
    found = entries.read_entries(LATE_SWAP_ENTRIES)
    values = {LATE_SWAP_IDS[2]: list("abcdefghi"), LATE_SWAP_IDS[0]: list("123456789")}
    written = rows(entries.upload_csv(found, values))
    assert written[0] == found.header
    assert [r[0] for r in written[1:]] == [LATE_SWAP_IDS[0], LATE_SWAP_IDS[2]]
    first = found.entries[0]
    assert written[1] == [first.entry_id, first.contest_name, first.contest_id,
                          first.entry_fee, *"123456789"]


# --- gui.py ---


def test_run_fills_carry_each_lineups_upload_row(classic_run, showdown_run):
    for run, path in ((classic_run, LATE_SWAP_ENTRIES), (showdown_run, SHOWDOWN_ENTRIES)):
        assert run.error is None, run.error
        fills = gui.run_fills(run, 1, path)
        assert list(fills) == [f"#{lu.number}" for lu in run.result.lineups]
        for fill, lineup_csv in zip(fills.values(), run.lineup_csvs, strict=True):
            upload_row = rows(lineup_csv)[-1]
            assert upload_row[1 : 1 + len(fill.values)] == list(fill.values)


def test_lineups_the_entries_file_cannot_match_cannot_fill(classic_run):
    # Read from the file given now: the Showdown pool lists none of these players.
    assert gui.run_fills(classic_run, 1, SHOWDOWN_ENTRIES) == {}
    assert gui.run_fills(classic_run, 1, None) == {}


def test_assign_edit_and_keep_picks_across_runs(classic_run):
    fills: gui.Fills = {}
    current = gui.run_fills(classic_run, 1, LATE_SWAP_ENTRIES)
    # Sequential, wrapping, and only into entries without a pick.
    fills[LATE_SWAP_IDS[1]] = current["#3"]
    assert gui.auto_assign(LATE_SWAP_IDS, [current["#1"], current["#2"]], fills) == 4
    assert {k: f.number for k, f in fills.items()} == {
        LATE_SWAP_IDS[1]: 3, LATE_SWAP_IDS[0]: 1, LATE_SWAP_IDS[2]: 2,
        LATE_SWAP_IDS[3]: 1, LATE_SWAP_IDS[4]: 2,
    }
    found = gui.read_fill_entries(CLASSIC, LATE_SWAP_ENTRIES)
    frame = gui.fill_entries_frame(CLASSIC, found, LATE_SWAP_ENTRIES, fills, 1)
    assert list(frame.columns) == [
        "Entry ID", "Contest", "Contest ID", "Fee", "Locked", "Current", gui.LINEUP, "Players",
    ]
    assert frame.at[LATE_SWAP_IDS[1], gui.LINEUP] == "#3"
    assert frame.at[LATE_SWAP_IDS[0], "Current"].startswith("Patrick Mahomes, Jahmyr Gibbs")

    # A second run: earlier picks keep their players under a run label.
    later = {label: gui.Fill(2, f.number, f.values, f.players) for label, f in current.items()}
    choices = gui.fill_choices(later, fills, 2)
    assert list(choices) == ["#1", "#2", "#3", "#1 (run 1)", "#2 (run 1)", "#3 (run 1)"]
    frame = gui.fill_entries_frame(CLASSIC, found, LATE_SWAP_ENTRIES, fills, 2)
    assert frame.at[LATE_SWAP_IDS[1], gui.LINEUP] == "#3 (run 1)"

    # Grid edits: a new pick, a cleared one, and a stale pending value (a no-op).
    edited = {0: {gui.LINEUP: "#2"}, 2: {gui.LINEUP: None}, 1: {gui.LINEUP: "#3 (run 1)"}}
    assert gui.apply_fill_edits(frame, edited, fills, choices, 2)
    assert fills[LATE_SWAP_IDS[0]].run == 2
    assert LATE_SWAP_IDS[2] not in fills
    assert fills[LATE_SWAP_IDS[1]].run == 1
    assert not gui.apply_fill_edits(frame, edited, fills, choices, 2)

    text = gui.fill_upload_csv(found, fills)
    written = rows(text)
    assert [r[0] for r in written[1:]] == [LATE_SWAP_IDS[i] for i in (0, 1, 3, 4)]
    assert written[1][4:] == list(fills[LATE_SWAP_IDS[0]].values)


def test_showdown_frame_has_no_locked_column(showdown_run):
    found = gui.read_fill_entries(SHOWDOWN, SHOWDOWN_ENTRIES)
    frame = gui.fill_entries_frame(SHOWDOWN, found, SHOWDOWN_ENTRIES, {}, 1)
    assert "Locked" not in frame
    assert set(frame["Current"]) == {"(empty)"}


def test_fill_file_name_tags_the_slate():
    assert gui.fill_file_name(CLASSIC, "DraftKings NFL DFS Projections -- Early Slate.csv").startswith(
        "upload-ready-DKEntries-early-"
    )
    assert gui.fill_file_name(SHOWDOWN, SHOWDOWN_FILE).startswith("upload-ready-DKEntries-2")


# --- The app, headless ---


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    """Downloads holding both projections files and the multi-entry Classic entries file."""
    folder = tmp_path / "Downloads"
    folder.mkdir()
    for path in (CLASSIC_FILE, SHOWDOWN_FILE, SHOWDOWN_ENTRIES, LATE_SWAP_ENTRIES):
        shutil.copy(path, folder)
    os.utime(folder / os.path.basename(LATE_SWAP_ENTRIES), (2e9, 2e9))
    monkeypatch.setattr(common, "DOWNLOADS_DIR", str(folder))
    monkeypatch.setattr(common, "EXPORT_DIR", str(tmp_path / "exports"))
    return folder


def run_app(**session) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=180)
    for key, value in session.items():
        app.session_state[key] = value
    app.run()
    assert not app.exception, app.exception
    return app


def click(app: AppTest, label: str, sidebar: bool = False) -> AppTest:
    buttons = app.sidebar.button if sidebar else app.button
    next(b for b in buttons if b.label == label).click().run()
    assert not app.exception, app.exception
    return app


def test_app_classic_fill_entries(downloads):
    app = run_app()
    assert not [h for h in app.subheader if h.value == "Fill entries"]
    app.number_input(key="w_Classic_num_lineups").set_value(2).run()
    app.number_input(key="w_Classic_min_uniques").set_value(2).run()
    click(app, "Optimize", sidebar=True)
    assert [h for h in app.subheader if h.value == "Fill entries"]
    click(app, "Auto-assign")
    fills = app.session_state["fills"][CLASSIC]
    assert [fills[i].number for i in LATE_SWAP_IDS] == [1, 2, 1, 2, 1]
    assert any("5 of 5 entries have a lineup." in c.value for c in app.caption)

    # Optimize again: the picks stay, labeled with their run.
    click(app, "Optimize", sidebar=True)
    assert app.session_state["fill_runs"][CLASSIC] == 2
    assert {f.run for f in app.session_state["fills"][CLASSIC].values()} == {1}
    click(app, "Clear all")
    assert app.session_state["fills"][CLASSIC] == {}


def test_app_fill_export_writes_on_download(downloads, tmp_path, classic_run):
    # The download button's callback can't be clicked headless; call the gui
    # helper it wraps.
    found = gui.read_fill_entries(CLASSIC, LATE_SWAP_ENTRIES)
    fills = {LATE_SWAP_IDS[0]: gui.run_fills(classic_run, 1, LATE_SWAP_ENTRIES)["#1"]}
    path = gui.write_fill_upload(gui.fill_upload_csv(found, fills), "upload.csv")
    assert os.path.dirname(path) == str(tmp_path / "exports")
    with open(path, encoding="utf-8") as handle:
        assert len(rows(handle.read())) == 2


def test_app_showdown_fill_and_wrong_format(downloads):
    app = run_app(w_format=SHOWDOWN)
    app.number_input(key="w_Showdown_num_lineups").set_value(2).run()
    app.number_input(key="w_Showdown_min_uniques").set_value(2).run()
    click(app, "Optimize", sidebar=True)
    # The newest entries file is the Classic one: the panel says so.
    assert any("Showdown lineups need" in e.value for e in app.error)
    app.selectbox(key="w_Showdown_entries").set_value(
        str(downloads / os.path.basename(SHOWDOWN_ENTRIES))
    ).run()
    click(app, "Auto-assign")
    assert len(app.session_state["fills"][SHOWDOWN]) == 2
    assert app.session_state["fills"][CLASSIC] == {}

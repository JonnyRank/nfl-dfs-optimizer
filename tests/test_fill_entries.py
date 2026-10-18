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
from nfl_dfs_optimizer.common import load_dk_name_ids
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
    found = entries.read_entries(LATE_SWAP_ENTRIES, gui.FILL_SLOTS[CLASSIC], CLASSIC)
    assert found.slot_labels == ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    assert found.header[:4] == ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
    assert [e.entry_id for e in found.entries][:5] == LATE_SWAP_IDS
    showdown_found = entries.read_entries(SHOWDOWN_ENTRIES, gui.FILL_SLOTS[SHOWDOWN], SHOWDOWN)
    assert showdown_found.slot_labels == ["CPT"] + ["FLEX"] * 5
    assert [e.cells for e in showdown_found.entries] == [[""] * 6, [""] * 6]


def test_read_entries_rejects_the_other_format(tmp_path):
    with pytest.raises(ValueError, match="has 6 roster slots .* Classic lineups need"):
        entries.read_entries(SHOWDOWN_ENTRIES, gui.FILL_SLOTS[CLASSIC], CLASSIC)
    # Nine slots in another order: the upload is positional, so it is refused too.
    with open(LATE_SWAP_ENTRIES, encoding="utf-8-sig") as handle:
        text = handle.read()
    shuffled = tmp_path / "DKEntriesShuffled.csv"
    shuffled.write_text(text.replace("QB,RB,RB,WR", "RB,QB,RB,WR", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="need an entries file with QB, RB, RB"):
        gui.read_fill_entries(CLASSIC, str(shuffled))
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
        fills = gui.run_fills(run, 1, load_dk_name_ids(path))
        assert list(fills) == [f"#{lu.number}" for lu in run.result.lineups]
        for fill, lineup_csv in zip(fills.values(), run.lineup_csvs, strict=True):
            upload_row = rows(lineup_csv)[-1]
            assert upload_row[1 : 1 + len(fill.values)] == list(fill.values)


def test_lineups_the_entries_file_cannot_match_cannot_fill(classic_run):
    # Read from the file given now: the Showdown pool lists none of these players.
    assert gui.run_fills(classic_run, 1, load_dk_name_ids(SHOWDOWN_ENTRIES)) == {}
    assert gui.run_fills(classic_run, 1, None) == {}


def test_showdown_players_mark_the_captain(showdown_run):
    fills = gui.run_fills(showdown_run, 1, load_dk_name_ids(SHOWDOWN_ENTRIES))
    captain = showdown_run.result.lineups[0].rows[0]["Player"].strip()
    assert fills["#1"].players.startswith(f"{captain} (CPT), ")


def test_assign_dialog_ticks_save_replace_and_remove(classic_run):
    found = gui.read_fill_entries(CLASSIC, LATE_SWAP_ENTRIES)
    current = gui.run_fills(classic_run, 1, load_dk_name_ids(LATE_SWAP_ENTRIES))
    one, two = current["#1"], current["#2"]
    fills: gui.Fills = {LATE_SWAP_IDS[0]: two}

    frame = gui.assign_frame(found, fills, {LATE_SWAP_IDS[1]}, 1)
    assert list(frame.columns) == [
        gui.ASSIGN, "Contest", "Contest ID", "Fee", "Entry ID", gui.ASSIGNED_TO,
    ]
    assert list(frame.index[:5]) == LATE_SWAP_IDS
    assert frame.at[LATE_SWAP_IDS[1], gui.ASSIGN]
    assert frame.at[LATE_SWAP_IDS[0], gui.ASSIGNED_TO] == "#2"
    assert frame.at[LATE_SWAP_IDS[1], gui.ASSIGNED_TO] == ""

    # Grid clicks: only values that differ from the ticks are new.
    ticked = {LATE_SWAP_IDS[1]}
    edited = {0: {gui.ASSIGN: True}, 1: {gui.ASSIGN: True}, 2: {gui.ASSIGN: False}}
    assert gui.apply_assign_edits(frame, edited, ticked)
    assert ticked == {LATE_SWAP_IDS[0], LATE_SWAP_IDS[1]}
    assert not gui.apply_assign_edits(frame, edited, ticked)

    # Saving: entry 0 swaps #2 for #1, entry 1 gets #1.
    assert gui.save_assignment(one, ticked, fills) == (2, 1, 0)
    assert fills == {LATE_SWAP_IDS[0]: one, LATE_SWAP_IDS[1]: one}
    assert gui.holders(one, fills) == {LATE_SWAP_IDS[0], LATE_SWAP_IDS[1]}
    assert gui.assignment_notice("#1", 2, 1, 0) == (
        "Lineup #1: assigned to 2 entries (1 replaced another lineup)."
    )
    contest = found.entries[0].contest_name
    summary = gui.assignment_summary(one, fills, found)
    assert summary.startswith("Assigned to 2 entries: ")
    assert contest in summary
    assert gui.assignment_summary(two, fills, found) == "Not assigned to an entry."

    # Unticking an entry the lineup fills takes it off; others keep theirs.
    fills[LATE_SWAP_IDS[2]] = two
    assert gui.save_assignment(one, {LATE_SWAP_IDS[1]}, fills) == (0, 0, 1)
    assert fills == {LATE_SWAP_IDS[1]: one, LATE_SWAP_IDS[2]: two}
    assert gui.assignment_notice("#1", 0, 0, 1) == "Lineup #1: removed from 1."
    assert gui.save_assignment(one, {LATE_SWAP_IDS[1]}, fills) == (0, 0, 0)
    assert gui.assignment_notice("#1", 0, 0, 0) == "Lineup #1: no changes."


def test_picks_keep_their_run_and_can_be_removed(classic_run):
    found = gui.read_fill_entries(CLASSIC, LATE_SWAP_ENTRIES)
    current = gui.run_fills(classic_run, 1, load_dk_name_ids(LATE_SWAP_ENTRIES))
    fills = {LATE_SWAP_IDS[3]: current["#3"], LATE_SWAP_IDS[1]: current["#1"]}
    fills["9999"] = current["#2"]  # an entry of another file

    # A second run: run 1's lineups keep their players under a run label, and
    # the same lineup number from the new run is a different lineup.
    later = gui.Fill(2, 3, current["#1"].values, current["#1"].players)
    assert gui.holders(later, fills) == set()
    frame = gui.assignments_frame(found, fills, 2)
    assert list(frame.columns) == [
        gui.REMOVE, "Contest", "Contest ID", "Fee", "Entry ID", gui.LINEUP, "Players",
    ]
    assert list(frame.index) == [LATE_SWAP_IDS[1], LATE_SWAP_IDS[3]]  # file order
    assert list(frame[gui.LINEUP]) == ["#1 (run 1)", "#3 (run 1)"]
    assert frame.at[LATE_SWAP_IDS[3], "Players"] == current["#3"].players
    assert gui.assign_frame(found, fills, set(), 2).at[LATE_SWAP_IDS[3], gui.ASSIGNED_TO] == (
        "#3 (run 1)"
    )

    text = gui.fill_upload_csv(found, fills)
    written = rows(text)
    assert [r[0] for r in written[1:]] == [LATE_SWAP_IDS[1], LATE_SWAP_IDS[3]]
    assert written[1][4:] == list(current["#1"].values)

    assert gui.apply_removals(frame, {1: {gui.REMOVE: True}, 0: {gui.REMOVE: False}}, fills) == 1
    assert set(fills) == {LATE_SWAP_IDS[1], "9999"}


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


def assign_buttons(app: AppTest) -> list:
    return [b for b in app.button if b.label == "Assign to entries"]


def test_app_classic_assign_from_the_cards(downloads):
    app = run_app()
    assert not [h for h in app.subheader if h.value == "Fill entries"]
    app.number_input(key="w_Classic_num_lineups").set_value(2).run()
    app.number_input(key="w_Classic_min_uniques").set_value(2).run()
    click(app, "Optimize", sidebar=True)
    assert [h for h in app.subheader if h.value == "Fill entries"]
    assert any("0 of 5 entries have a lineup." in c.value for c in app.caption)
    buttons = assign_buttons(app)
    assert len(buttons) == 2 and not any(b.disabled for b in buttons)
    # No Locked or Current: that is late swap's business.
    assert not [b for b in app.button if b.label == "Auto-assign"]

    # The dialog opens over the page, naming the lineup. (Its clicks rerun
    # only the dialog in a browser; AppTest reruns the page, so saving is
    # covered through gui.save_assignment().)
    buttons[0].click().run()
    assert not app.exception, app.exception
    assert any(m.value.startswith("**Lineup #1:**") for m in app.markdown)
    assert [b for b in app.button if b.label == "Save"]
    assert app.session_state["assign_ticks"] == set()

    # Picks made in the dialog show on the card and in the panel.
    run = app.session_state["results"][CLASSIC]
    current = gui.run_fills(run, 1, load_dk_name_ids(LATE_SWAP_ENTRIES))
    app.session_state["fills"][CLASSIC].update(
        {LATE_SWAP_IDS[0]: current["#1"], LATE_SWAP_IDS[1]: current["#1"]}
    )
    app.run()
    assert any(c.value.startswith("Assigned to 2 entries: ") for c in app.caption)
    assert any("2 of 5 entries have a lineup." in c.value for c in app.caption)
    assert [e for e in app.expander if e.label == "Assignments (2)"]
    # Reopening the dialog starts from the entries the lineup fills.
    assign_buttons(app)[0].click().run()
    assert app.session_state["assign_ticks"] == {LATE_SWAP_IDS[0], LATE_SWAP_IDS[1]}

    # Optimize again: the picks stay, labeled with their run.
    click(app, "Optimize", sidebar=True)
    assert app.session_state["fill_runs"][CLASSIC] == 2
    assert {f.run for f in app.session_state["fills"][CLASSIC].values()} == {1}
    assert any("2 of 5 entries have a lineup." in c.value for c in app.caption)
    click(app, "Clear all")
    assert app.session_state["fills"][CLASSIC] == {}


def test_app_fill_export_writes_on_download(downloads, tmp_path, classic_run):
    # The download button's callback can't be clicked headless; call the gui
    # helper it wraps.
    found = gui.read_fill_entries(CLASSIC, LATE_SWAP_ENTRIES)
    fills = {LATE_SWAP_IDS[0]: gui.run_fills(classic_run, 1, load_dk_name_ids(LATE_SWAP_ENTRIES))["#1"]}
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
    assert all(b.disabled for b in assign_buttons(app))
    app.selectbox(key="w_Showdown_entries").set_value(
        str(downloads / os.path.basename(SHOWDOWN_ENTRIES))
    ).run()
    assert not any(b.disabled for b in assign_buttons(app))
    assign_buttons(app)[1].click().run()
    assert not app.exception, app.exception
    assert any(m.value.startswith("**Lineup #2:**") and "(CPT)" in m.value for m in app.markdown)

"""
Late swap outside the CLI: the core's run() against the -ls goldens, entry
selection, and the app's Late swap page (gui.py directly, and app.py headless).
The clock is frozen as the parity harness freezes it.
"""

import os
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from parity_harness import (
    CLASSIC_FILE,
    GOLDEN,
    LATE_SWAP_ENTRIES,
    frozen_datetime,
)
from streamlit.testing.v1 import AppTest

from nfl_dfs_optimizer import classic, common, gui, late_swap
from nfl_dfs_optimizer.gui import LATE_SWAP
from nfl_dfs_optimizer.late_swap import LateSwapOptions

APP = os.path.join(os.path.dirname(gui.__file__), "app.py")
EASTERN = ZoneInfo("America/New_York")
AT_1_30 = datetime(2026, 9, 27, 13, 30, tzinfo=EASTERN)


@pytest.fixture(scope="module")
def classic_df():
    return classic.with_ownership(
        classic.load_player_data(CLASSIC_FILE).df, classic.OWNERSHIP_LARGE_FIELD
    )


def golden_upload(name: str) -> str:
    """A late-swap golden's upload file, without the harness's file-name line."""
    path = os.path.join(GOLDEN, "classic", f"{name}.export.csv")
    with open(path, encoding="utf-8") as handle:
        return handle.read().split("\n", 1)[1]


def entry_ids() -> list[str]:
    swap = late_swap.load_slate(classic.load_player_data(CLASSIC_FILE).df, LATE_SWAP_ENTRIES)
    return [entry.entry_id for entry in swap.entries]


# --- The core ---


def test_run_matches_the_cli_golden_and_is_silent(classic_df, capsys):
    options = LateSwapOptions(min_uniques=2, min_salary=49500, stack=1)
    result = late_swap.run(classic_df, LATE_SWAP_ENTRIES, options, now=AT_1_30)
    text = late_swap.upload_csv(result.slate, result.outcomes).replace("\r\n", "\n")
    assert text == golden_upload("late_swap_1pm")
    assert result.summary.startswith("Late swap complete:")
    assert capsys.readouterr() == ("", "")


def test_selected_entries_only_are_rebuilt_and_written(classic_df):
    ids = entry_ids()
    chosen = frozenset(ids[1:3])
    options = LateSwapOptions(min_uniques=2, entry_ids=chosen)
    result = late_swap.run(classic_df, LATE_SWAP_ENTRIES, options, now=AT_1_30)
    assert [o.entry.entry_id for o in result.outcomes] == ids[1:3]
    assert [o.number for o in result.outcomes] == [1, 2]
    rows = late_swap.upload_csv(result.slate, result.outcomes).splitlines()
    assert len(rows) == 1 + 2


def test_skipped_entries_still_count_toward_min_uniques(classic_df):
    # Skip the first entry: the second, in the same contest, must still differ
    # from it by -u players, even though it comes later in the file.
    ids = entry_ids()
    options = LateSwapOptions(min_uniques=9, entry_ids=frozenset([ids[1]]))
    result = late_swap.run(classic_df, LATE_SWAP_ENTRIES, options, now=AT_1_30)
    skipped = late_swap.split_entry_slots(result.slate.entries[0], result.slate.pool)[0]
    rebuilt = result.outcomes[0]
    locked = {rebuilt.final_ids[i] for i in rebuilt.locked}
    shared = (set(rebuilt.final_ids) & set(skipped)) - locked
    assert not shared or "could not differ" in rebuilt.note


def test_unknown_or_empty_entry_selection_raises(classic_df):
    with pytest.raises(ValueError, match="No entries are selected"):
        late_swap.run(classic_df, LATE_SWAP_ENTRIES, LateSwapOptions(entry_ids=frozenset()))
    with pytest.raises(ValueError, match="has no entry 123"):
        late_swap.run(
            classic_df, LATE_SWAP_ENTRIES, LateSwapOptions(entry_ids=frozenset(["123"]))
        )


def test_options_validate_like_the_strict_classic_checks():
    with pytest.raises(ValueError, match="--min-uniques"):
        LateSwapOptions(min_uniques=10).validate()
    with pytest.raises(ValueError, match="salary cap"):
        LateSwapOptions(min_salary=60000).validate()


# --- gui.py ---


def test_entries_frame_and_swap_clicks():
    frame = gui.entries_frame(LATE_SWAP_ENTRIES, set(), now=AT_1_30)
    assert list(frame.columns) == [
        gui.SWAP, "Entry ID", "Contest", "Contest ID", "Fee", "Locked", "Lineup",
    ]
    labels = gui.contest_labels(frame)
    assert list(labels) == ["195905123", "195905999"]
    assert labels["195905999"] == "NFL Parity Contest 195905999 (195905999)"
    assert frame[gui.SWAP].all()
    assert frame["Locked"].str.endswith("/9").all()
    assert frame.iloc[0]["Lineup"].startswith("Patrick Mahomes")

    skipped: set[str] = set()
    shown = gui.filter_entries(frame, [frame.iloc[2]["Contest ID"]])
    assert set(shown["Contest ID"]) == {frame.iloc[2]["Contest ID"]}
    assert gui.apply_entry_edits(shown, {0: {gui.SWAP: False}}, skipped)
    assert skipped == {shown.index[0]}
    # The same pending value again is not new.
    assert not gui.apply_entry_edits(shown, {0: {gui.SWAP: False}}, skipped)
    assert gui.apply_entry_edits(shown, {0: {gui.SWAP: True}}, skipped)
    assert skipped == set()


def test_validation_needs_an_entries_file(tmp_path):
    settings = gui.Settings()
    assert gui.validation_errors(LATE_SWAP, settings) == [
        "Late swap needs a DraftKings entries file."
    ]
    settings.dk_entries = str(tmp_path / "missing.csv")
    assert "not found" in gui.validation_errors(LATE_SWAP, settings)[0]
    settings.dk_entries = LATE_SWAP_ENTRIES
    settings.min_uniques = 10
    assert "--min-uniques" in gui.validation_errors(LATE_SWAP, settings)[0]


def test_gui_run_writes_only_under_export(tmp_path, monkeypatch):
    downloads, exports = tmp_path / "Downloads", tmp_path / "exports"
    downloads.mkdir()
    monkeypatch.setattr(common, "DOWNLOADS_DIR", str(downloads))
    monkeypatch.setattr(common, "EXPORT_DIR", str(exports))
    df = classic.load_player_data(CLASSIC_FILE).df
    settings = gui.Settings(min_uniques=2, min_salary=49500, stack=1, dk_entries=LATE_SWAP_ENTRIES)
    run = gui.run_late_swap(
        df, CLASSIC_FILE, settings, {}, {}, None, entry_ids=None, now=AT_1_30
    )
    assert run.error is None, run.error
    assert run.export_path is None
    assert not os.listdir(downloads) and not exports.exists()
    assert run.upload_csv.replace("\r\n", "\n") == golden_upload("late_swap_1pm")
    assert run.file_name.startswith("upload-ready-DKEntries-2")

    settings.export = True
    run = gui.run_late_swap(
        df, CLASSIC_FILE, settings, {}, {}, None, entry_ids=None, now=AT_1_30
    )
    assert os.path.dirname(run.export_path) == str(exports)
    with open(run.export_path, encoding="utf-8") as handle:
        assert handle.read() == golden_upload("late_swap_1pm")
    assert not os.listdir(downloads)
    table = gui.swap_table(run.result, run.result.outcomes[1])
    assert list(table.columns) == [
        "Slot", "Player", "Pos", "Team", "Salary", "Proj", "Own%", "Ceiling", "Status",
    ]
    assert "→" in gui.swap_totals(run.result, run.result.outcomes[1])


def test_gui_run_applies_locks_excludes_and_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DOWNLOADS_DIR", str(tmp_path))
    df = classic.load_player_data(CLASSIC_FILE).df
    settings = gui.Settings(dk_entries=LATE_SWAP_ENTRIES)
    base = gui.run_late_swap(df, CLASSIC_FILE, settings, {}, {}, None, None, now=AT_1_30)
    first = base.result.outcomes[1]
    new_ids = [
        dk_id for idx, dk_id in enumerate(first.final_ids)
        if idx not in first.locked and dk_id is not None
    ]
    excluded = {new_ids[0]: gui.SLOT_ANY}
    run = gui.run_late_swap(df, CLASSIC_FILE, settings, {}, excluded, None, None, now=AT_1_30)
    assert all(new_ids[0] not in o.final_ids or new_ids[0] in
               [o.original_ids[i] for i in o.locked] for o in run.result.outcomes)

    # An edit that makes an unstarted bench player irresistible puts him in.
    unstarted = [
        p for p in base.result.slate.candidates(set()).values()
        if p.primary_slot == "WR" and p.dk_id not in first.final_ids
    ]
    star = min(unstarted, key=lambda p: p.salary).dk_id
    edits = {star: {"Projection": 500.0}}
    run = gui.run_late_swap(df, CLASSIC_FILE, settings, {}, {}, edits, None, now=AT_1_30)
    assert star in run.result.outcomes[1].final_ids
    assert run.edited == {star}
    assert gui.EDITED_MARK in "".join(gui.swap_table(run.result, run.result.outcomes[1], run.edited)["Player"])


def test_gui_run_turns_errors_into_messages(classic_df, tmp_path):
    run = gui.run_late_swap(
        classic_df, CLASSIC_FILE, gui.Settings(dk_entries=str(tmp_path / "nope.csv")),
        {}, {}, None, None,
    )
    assert run.error and run.result is None


# --- The app, headless ---


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    """Downloads holding the projections and the filled-in entries file, clock at 1:30PM."""
    folder = tmp_path / "Downloads"
    folder.mkdir()
    for path in (CLASSIC_FILE, LATE_SWAP_ENTRIES):
        shutil.copy(path, folder)
    monkeypatch.setattr(common, "DOWNLOADS_DIR", str(folder))
    monkeypatch.setattr(common, "EXPORT_DIR", str(tmp_path / "exports"))
    frozen = frozen_datetime("2026-09-27 13:30")
    monkeypatch.setattr(late_swap, "datetime", frozen)
    monkeypatch.setattr(gui, "datetime", frozen)
    return folder


def run_app(**session) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=180)
    for key, value in session.items():
        app.session_state[key] = value
    app.run()
    assert not app.exception, app.exception
    return app


def run_button(app: AppTest):
    return next(b for b in app.button if b.label == "Run late swap")


def test_app_late_swap_page_runs_every_entry_by_default(downloads):
    app = run_app(w_format=LATE_SWAP)
    assert app.number_input(key="w_Late swap_min_uniques").value == 1
    assert not [n for n in app.number_input if n.key == "w_Late swap_num_lineups"]
    app.number_input(key="w_Late swap_min_uniques").set_value(2).run()
    app.number_input(key="w_Late swap_min_salary").set_value(49500).run()
    app.number_input(key="w_Late swap_stack").set_value(1).run()
    run_button(app).click().run()
    assert not app.exception, app.exception
    run = app.session_state["results"][LATE_SWAP]
    assert run.error is None, run.error
    assert run.upload_csv.replace("\r\n", "\n") == golden_upload("late_swap_1pm")
    # Without Export to CSV a run writes nothing.
    assert sorted(os.listdir(downloads)) == sorted(
        os.path.basename(p) for p in (CLASSIC_FILE, LATE_SWAP_ENTRIES)
    )
    assert len(app.dataframe) == 2 + 5  # player grid, entries grid, one table per entry
    assert app.get("download_button")


def test_app_late_swap_export_toggle(downloads, tmp_path):
    app = run_app(w_format=LATE_SWAP)
    app.toggle(key="w_Late swap_export").set_value(True).run()
    run_button(app).click().run()
    run = app.session_state["results"][LATE_SWAP]
    assert os.path.dirname(run.export_path) == str(tmp_path / "exports")


def test_app_late_swap_skips_unselected_entries(downloads):
    ids = entry_ids()
    app = run_app(w_format=LATE_SWAP, skipped_entries={ids[0], ids[4]})
    assert any("3 of 5 entries selected" in c.value for c in app.caption)
    run_button(app).click().run()
    run = app.session_state["results"][LATE_SWAP]
    assert [o.entry.entry_id for o in run.result.outcomes] == ids[1:4]


def test_app_late_swap_blocks_a_run_with_nothing_selected(downloads):
    app = run_app(w_format=LATE_SWAP, skipped_entries=set(entry_ids()))
    assert run_button(app).disabled
    next(b for b in app.button if b.label == "Select all shown").click().run()
    assert not run_button(app).disabled


def test_app_late_swap_keeps_its_own_settings_and_locks(downloads):
    app = run_app()
    app.number_input(key="w_Classic_min_uniques").set_value(3).run()
    app.radio(key="w_format").set_value(LATE_SWAP).run()
    assert app.number_input(key="w_Late swap_min_uniques").value == 1
    assert app.session_state["selections"][LATE_SWAP] is not app.session_state["selections"][
        gui.CLASSIC
    ]


def test_a_failed_export_keeps_the_swap(tmp_path, monkeypatch):
    # The export drive offline: EXPORT_DIR sits under a file, so makedirs fails.
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("")
    monkeypatch.setattr(common, "EXPORT_DIR", str(blocker / "exports"))
    df = classic.load_player_data(CLASSIC_FILE).df
    settings = gui.Settings(export=True, dk_entries=LATE_SWAP_ENTRIES)
    run = gui.run_late_swap(df, CLASSIC_FILE, settings, {}, {}, None, None, now=AT_1_30)
    assert run.error is None
    assert run.export_path is None
    assert run.export_error.startswith("Export failed:")
    assert run.result.outcomes and run.upload_csv

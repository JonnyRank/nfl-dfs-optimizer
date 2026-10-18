"""
Hosted mode (NFL_DFS_HOSTED=true, as on Streamlit Community Cloud): the mode
switch, saving uploads per session, the "Download all lineups" file, and the
app headless with uploads seeded through session_state (AppTest can't drive
st.file_uploader).
"""

import io
import os
import re
import shutil

import pandas as pd
import pytest
from parity_harness import (
    CLASSIC_ENTRIES,
    CLASSIC_FILE,
    SHOWDOWN_ENTRIES,
    SHOWDOWN_FILE,
)
from streamlit.testing.v1 import AppTest
from test_gui import (  # noqa: F401 -- downloads is a fixture
    click_optimize,
    downloads,
    golden_scores,
    run_app,
    scores,
)

from nfl_dfs_optimizer import classic, common, gui
from nfl_dfs_optimizer.gui import CLASSIC, LATE_SWAP, SHOWDOWN

# --- The mode switch ---


@pytest.mark.parametrize(
    ("value", "hosted"),
    [(None, False), ("", False), ("true", True), ("TRUE", True), (" true ", True),
     ("1", False), ("false", False), ("yes", False)],
)
def test_hosted_mode_reads_the_environment(monkeypatch, value, hosted):
    if value is None:
        monkeypatch.delenv(gui.HOSTED_ENV, raising=False)
    else:
        monkeypatch.setenv(gui.HOSTED_ENV, value)
    assert gui.hosted_mode() is hosted


CLOUD_PATH = "/mount/src/nfl-dfs-optimizer/src/nfl_dfs_optimizer/gui.py"


def test_community_cloud_is_hosted_without_the_secret(monkeypatch):
    # Fail safe: a missing or mistyped secret on Community Cloud must not
    # leave the public app in local mode (server path box, export toggle).
    monkeypatch.delenv(gui.HOSTED_ENV, raising=False)
    assert gui.on_community_cloud(CLOUD_PATH)
    assert gui.hosted_mode(CLOUD_PATH)
    monkeypatch.setenv(gui.HOSTED_ENV, "flase")
    assert gui.hosted_mode(CLOUD_PATH)


def test_this_checkout_is_not_community_cloud():
    assert not gui.on_community_cloud()
    assert not gui.on_community_cloud(r"C:\Users\someone\mount\src\app.py")


# --- Saving uploads ---


@pytest.mark.parametrize(
    ("name", "saved"),
    [
        ("DraftKings NFL DFS Projections -- Early Slate.csv",
         "DraftKings NFL DFS Projections -- Early Slate.csv"),
        (r"C:\Users\someone\Downloads\DKEntries (1).csv", "DKEntries (1).csv"),
        ("../../etc/passwd", "passwd"),
        ("we<ird>:name?.csv", "we_ird__name_.csv"),
        ("..", gui.UPLOAD_NAME_DEFAULT),
        ("", gui.UPLOAD_NAME_DEFAULT),
        (None, gui.UPLOAD_NAME_DEFAULT),
    ],
)
def test_safe_file_name(name, saved):
    assert gui.safe_file_name(name) == saved


def test_uploaded_name_keeps_the_slate(tmp_path):
    path = gui.save_upload(
        str(tmp_path), "Classic-projections",
        "DraftKings NFL DFS Projections -- Late Slate.csv", b"x",
    )
    assert classic.detect_slate(path) == "Late"


def test_uploads_land_in_separate_per_session_folders():
    first, second = gui.new_upload_folder(), gui.new_upload_folder()
    try:
        assert first != second
        slot = gui.upload_slot(CLASSIC, gui.PROJECTIONS_UPLOAD)
        name = "DraftKings NFL DFS Projections -- Main Slate.csv"
        a = gui.save_upload(first, slot, name, b"user a")
        b = gui.save_upload(second, slot, name, b"user b")
        assert a != b
        assert os.path.basename(a) == os.path.basename(b) == name
        with open(a, "rb") as handle_a, open(b, "rb") as handle_b:
            assert (handle_a.read(), handle_b.read()) == (b"user a", b"user b")
    finally:
        shutil.rmtree(first)
        shutil.rmtree(second)


def test_a_new_upload_replaces_the_old_one(tmp_path):
    folder = str(tmp_path)
    slot = gui.upload_slot(CLASSIC, gui.ENTRIES_UPLOAD)
    old = gui.save_upload(folder, slot, "DKEntries.csv", b"old")
    new = gui.save_upload(folder, slot, "DKEntries (1).csv", b"new")
    assert not os.path.exists(old)
    assert os.listdir(os.path.dirname(new)) == ["DKEntries (1).csv"]
    # Same name in another slot: no collision.
    other = gui.save_upload(folder, gui.upload_slot(SHOWDOWN, gui.ENTRIES_UPLOAD), "DKEntries (1).csv", b"sd")
    assert other != new and os.path.exists(new)


class FakeUpload:
    """What the uploader callback reads from Streamlit's UploadedFile."""

    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


def test_store_upload_saves_then_forgets_a_cleared_upload(tmp_path):
    uploads = {}
    slot = gui.upload_slot(CLASSIC, gui.PROJECTIONS_UPLOAD)
    gui.store_upload(uploads, str(tmp_path), slot, FakeUpload("a/Early Slate.csv", b"one"))
    path = uploads[slot]
    assert os.path.basename(path) == "Early Slate.csv"
    gui.store_upload(uploads, str(tmp_path), slot, FakeUpload("Late Slate.csv", b"two"))
    assert not os.path.exists(path) and os.path.basename(uploads[slot]) == "Late Slate.csv"
    gui.store_upload(uploads, str(tmp_path), slot, None)
    assert uploads == {}


def test_stale_upload_folders_are_removed(tmp_path):
    now = 1_000_000_000.0
    day = gui.UPLOAD_MAX_AGE_SECONDS

    def folder(name, age):
        path = tmp_path / name
        (path / "Classic-entries").mkdir(parents=True)
        for touched in (path / "Classic-entries", path):
            os.utime(touched, (now - age, now - age))
        return path

    stale = folder("nfl-dfs-old", day + 60)
    fresh = folder("nfl-dfs-new", 60)
    other = folder("someone-else", day * 3)
    # A slot written recently keeps an old session folder alive.
    active = folder("nfl-dfs-active", day + 60)
    os.utime(active / "Classic-entries", (now - 60, now - 60))

    assert gui.remove_stale_upload_folders(str(tmp_path), now=now) == [str(stale)]
    assert fresh.exists() and other.exists() and active.exists()
    assert gui.remove_stale_upload_folders(str(tmp_path / "missing"), now=now) == []


# --- Download all lineups ---


def _read(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


@pytest.mark.parametrize(
    ("fmt", "projections", "entries", "target"),
    [
        (CLASSIC, CLASSIC_FILE, CLASSIC_ENTRIES, common.TARGET_PROJECTION),
        (SHOWDOWN, SHOWDOWN_FILE, SHOWDOWN_ENTRIES, common.TARGET_BLEND),
        (SHOWDOWN, SHOWDOWN_FILE, None, common.TARGET_PROJECTION),  # no upload rows
    ],
)
def test_download_all_matches_the_export(tmp_path, monkeypatch, fmt, projections, entries, target):
    monkeypatch.setattr(common, "EXPORT_DIR", str(tmp_path))
    df = gui.load_pool(fmt, projections).df
    settings = gui.Settings(num_lineups=3, min_uniques=2, export=True, dk_entries=entries, target=target)
    run = gui.optimize(fmt, df, projections, settings, {}, {})
    assert run.export_path
    assert gui.all_lineups_csv(run, entries) == _read(run.export_path)
    # The name is -e's, timestamp aside.
    stamp = re.compile(r"_\d[\d_-]*\.csv$")
    exported_name = os.path.basename(run.export_path)
    assert stamp.sub("", gui.all_lineups_file_name(run)) == stamp.sub("", exported_name)


def test_download_all_without_lineups_is_empty():
    assert gui.all_lineups_csv(gui.RunOutcome(CLASSIC, common.TARGET_PROJECTION), None) == ""


# --- The app, hosted ---


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv(gui.HOSTED_ENV, "true")


@pytest.fixture
def session_uploads(tmp_path):
    """A session's upload folder holding the public fixtures, as the uploader would save them."""
    folder = tmp_path / "session"
    uploads = {}
    for fmt, kind, source in (
        (CLASSIC, gui.PROJECTIONS_UPLOAD, CLASSIC_FILE),
        (CLASSIC, gui.ENTRIES_UPLOAD, CLASSIC_ENTRIES),
        (SHOWDOWN, gui.PROJECTIONS_UPLOAD, SHOWDOWN_FILE),
        (SHOWDOWN, gui.ENTRIES_UPLOAD, SHOWDOWN_ENTRIES),
    ):
        slot = gui.upload_slot(fmt, kind)
        with open(source, "rb") as handle:
            uploads[slot] = gui.save_upload(str(folder), slot, os.path.basename(source), handle.read())
    return {"upload_dir": str(folder), "uploads": uploads}


def _uploaders(app: AppTest) -> list:
    return app.get("file_uploader")


@pytest.mark.parametrize("page", [CLASSIC, SHOWDOWN, LATE_SWAP])
def test_hosted_hides_pickers_browse_and_export(hosted, downloads, page):  # noqa: F811
    app = run_app(w_format=page, w_Classic_file="__other__")
    fmt = gui.page_format(page)
    assert len(_uploaders(app)) == 2
    assert not [s for s in app.selectbox if s.key in (f"w_{page}_file", f"w_{page}_entries")]
    assert not [b for b in app.button if b.label == "Browse..."]
    assert not app.text_input  # no path box
    assert not app.toggle  # no Export to CSV
    assert f"Upload a {gui.PROJECTIONS_GLOBS[fmt]} file" in app.info[0].value
    assert os.path.isdir(app.session_state["upload_dir"])


def test_hosted_upload_folder_is_kept_for_the_session(hosted, downloads):  # noqa: F811
    app = run_app()
    folder = app.session_state["upload_dir"]
    app.radio(key="w_format").set_value(SHOWDOWN).run()
    assert app.session_state["upload_dir"] == folder


def test_hosted_optimize_offers_download_all(hosted, downloads, session_uploads, tmp_path):  # noqa: F811
    app = run_app(**session_uploads)
    assert any(
        c.value == "Using DraftKings NFL DFS Projections -- Main Slate.csv (uploaded)"
        for c in app.sidebar.caption
    )
    app.number_input(key="w_Classic_num_lineups").set_value(3).run()
    app.number_input(key="w_Classic_min_uniques").set_value(2).run()
    click_optimize(app)
    run = app.session_state["results"][CLASSIC]
    assert scores(run.result) == golden_scores("classic", "multi_u2")[:3]
    assert run.export_path is None
    assert not os.path.exists(tmp_path / "exports")  # EXPORT_DIR is never touched
    labels = [b.proto.label for b in app.get("download_button")]
    # Filled entries sits above the lineups.
    assert labels == ["Download filled entries CSV", "Download all lineups"] + [
        "Download CSV"
    ] * 3

    # The file matches what -e writes for the same run.
    name, text = app.session_state["all_lineups"][CLASSIC]
    assert name.startswith("nfl_classic_multi_lineups_")
    entries = session_uploads["uploads"][gui.upload_slot(CLASSIC, gui.ENTRIES_UPLOAD)]
    assert text == gui.all_lineups_csv(run, entries)
    exported = pd.read_csv(io.StringIO(text))
    assert len(exported) == 3 * 11  # 9 players, TOTAL, upload row


def test_hosted_late_swap_writes_nothing(hosted, downloads, session_uploads, tmp_path):  # noqa: F811
    app = run_app(w_format=LATE_SWAP, **session_uploads)
    assert not app.toggle
    assert not app.exception
    assert len(_uploaders(app)) == 2


# --- Local mode is unchanged ---


def test_local_mode_has_no_uploads_or_download_all(monkeypatch, downloads):  # noqa: F811
    monkeypatch.delenv(gui.HOSTED_ENV, raising=False)
    app = run_app()
    assert not _uploaders(app)
    assert "upload_dir" not in app.session_state and "all_lineups" not in app.session_state
    assert app.selectbox(key="w_Classic_file").value == "__newest__"
    assert app.toggle(key="w_Classic_export").value is False
    click_optimize(app)
    labels = [b.proto.label for b in app.get("download_button")]
    assert labels == ["Download filled entries CSV", "Download CSV"]


def test_hosted_flag_set_to_anything_else_is_local(monkeypatch, downloads):  # noqa: F811
    monkeypatch.setenv(gui.HOSTED_ENV, "1")
    app = run_app()
    assert not _uploaders(app)
    assert app.toggle(key="w_Classic_export")

"""
Streamlit app for the Classic and Showdown optimizers and Classic late swap.

Run it with the launcher ("NFL DFS Optimizer.bat" in the repo root) or:

    uv run streamlit run src/nfl_dfs_optimizer/app.py

The logic lives in gui.py; this file only draws widgets and keeps state.
Projection / Ceiling / ownership edits live in st.session_state["edits"] the
same way locks do, in memory only; the projections file is never written.
Widget keys start with "w_" and are re-assigned at the top of every run, so a
value survives while its widget is hidden (the other format's settings). Locks
and excludes live in st.session_state["selections"], keyed by player, not in
the grid, so they survive filtering, sorting, and a re-downloaded file.
Late swap is a third page over the Classic projections; locks, edits,
settings and results are all kept per page. The entries it skips are held in
st.session_state["skipped_entries"] by Entry ID, so every entry of a newly
downloaded file starts selected.
Each lineup card's "Assign to entries" dialog gives the lineup to DraftKings
entries; Fill entries, above the lineups, lists the picks and downloads the
upload file. The picks live in st.session_state["fills"] per format and
outlive later Optimize runs.

Hosted mode (gui.hosted_mode(): NFL_DFS_HOSTED=true, as on Community Cloud)
swaps the Downloads pickers for uploads, saved per session under
st.session_state["upload_dir"] and listed in st.session_state["uploads"] by
slot, hides Export to CSV (nothing is written outside that folder), and adds
a "Download all lineups" button holding the -e file. Unset, the app is the
local one, unchanged.
"""

import os

import streamlit as st

from nfl_dfs_optimizer import classic, common, gui
from nfl_dfs_optimizer.common import SALARY_CAP, PlayerDataError
from nfl_dfs_optimizer.gui import CLASSIC, EXCLUDE, LATE_SWAP, LOCK, SHOWDOWN

NEWEST = "__newest__"
OTHER = "__other__"
NONE = "__none__"
DESCENDING = "High to low"
ASCENDING = "Low to high"

st.set_page_config(page_title="NFL DFS Optimizer", layout="wide")

state = st.session_state
# Streamlit drops a widget's value on any run that doesn't draw it; assigning
# it back keeps the hidden format's settings.
for widget_key in list(state.keys()):
    if isinstance(widget_key, str) and widget_key.startswith("w_"):
        state[widget_key] = state[widget_key]
state.setdefault("selections", {})
state.setdefault("edits", {})
for app_page in gui.PAGES:
    state.selections.setdefault(app_page, {"locks": {}, "excludes": {}})
    state.edits.setdefault(app_page, {})
state.setdefault("grid_version", 0)
state.setdefault("results", {})
state.setdefault("notices", [])
state.setdefault("skipped_entries", set())
state.setdefault("entries_version", 0)
# Classic/Showdown entry picks: {Entry ID: gui.Fill} per format, the
# format's Optimize count (a Fill's run), the assignments table's version,
# and the open assign dialog's ticked Entry IDs and grid version.
state.setdefault("fills", {fill_format: {} for fill_format in gui.FORMATS})
state.setdefault("fill_runs", {fill_format: 0 for fill_format in gui.FORMATS})
state.setdefault("fills_version", 0)
state.setdefault("fill_notices", [])
state.setdefault("assign_ticks", set())
state.setdefault("assign_version", 0)
hosted = gui.hosted_mode()
if hosted:
    if "upload_dir" not in state:
        state.upload_dir = gui.new_upload_folder()
    state.setdefault("uploads", {})
    state.setdefault("all_lineups", {})


@st.cache_data(show_spinner="Loading projections...", max_entries=8)
def cached_pool(fmt: str, path: str, modified: float):
    """The loaded file; `modified` makes a re-saved file load again."""
    return gui.load_pool(fmt, path)


# Fill entries reads the entries file on every rerun (each Search keystroke
# among them), so each read is cached on the file's mtime.
@st.cache_data(show_spinner=False, max_entries=8)
def cached_fill_entries(fill_format: str, path: str, modified: float):
    return gui.read_fill_entries(fill_format, path)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_name_ids(path: str, modified: float):
    return common.load_dk_name_ids(path)


def widget(name: str, default):
    """The session key for this page's `name` widget, seeded with `default`."""
    key = f"w_{page}_{name}"
    state.setdefault(key, default)
    return key


def pick_file(
    label: str,
    name: str,
    files: list[str],
    newest_label: str,
    allow_none: bool,
    help_text: str | None = None,
):
    """A file chooser: newest match (default), a specific match, none, or a typed path."""
    options = [NEWEST, *([NONE] if allow_none else []), *files, OTHER]
    key = widget(name, NEWEST)
    if state[key] not in options:
        state[key] = NEWEST
    labels = {NEWEST: newest_label, NONE: "None", OTHER: "Other file..."}
    choice = st.selectbox(
        label,
        options,
        key=key,
        format_func=lambda o: labels.get(o, os.path.basename(o)),
        help=help_text,
    )
    if choice == NEWEST:
        path = files[0] if files else None
    elif choice == NONE:
        path = None
    elif choice == OTHER:
        path_key = widget(f"{name}_other", "")
        typed_col, browse_col = st.columns([4, 1], vertical_alignment="bottom")
        typed = typed_col.text_input(f"{label} path", key=path_key)
        browse_col.button(
            "Browse...",
            key=f"browse_{page}_{name}",
            on_click=browse_into,
            args=(path_key, f"Choose the {label.lower()} file"),
            help="Pick the file in a Windows Open dialog.",
        )
        path = typed.strip().strip('"') or None
    else:
        path = choice
    if path and os.path.isfile(path):
        st.caption(gui.describe_file(path))
    elif path:
        st.caption(f"Not found: {path}")
    elif choice == NEWEST:
        st.caption("No matching file in Downloads.")
    return path


def browse_into(path_key: str, title: str) -> None:
    """Button callback: fill a path box from the Open dialog (cancel keeps it)."""
    current = state.get(path_key, "").strip().strip('"')
    try:
        chosen = gui.browse_for_csv(title, os.path.dirname(current) if current else None)
    except RuntimeError as exc:
        state.notices.append(str(exc))
        return
    if chosen:
        state[path_key] = chosen


def upload_file(label: str, kind: str, help_text: str) -> str | None:
    """Hosted: an uploader for this format's `kind` file; returns the saved file's path."""
    slot = gui.upload_slot(fmt, kind)
    # Not a "w_" key: an uploader's value can't be assigned back.
    st.file_uploader(
        label, type="csv", key=f"upload_{slot}", on_change=store_upload, args=(slot,),
        help=help_text,
    )
    path = state.uploads.get(slot)
    if path and os.path.isfile(path):
        st.caption(f"Using {os.path.basename(path)} (uploaded)")
        return path
    st.caption("No file uploaded.")
    return None


def store_upload(slot: str) -> None:
    """Uploader callback: save a new upload over the slot's old one, or forget a cleared one."""
    gui.store_upload(state.uploads, state.upload_dir, slot, state.get(f"upload_{slot}"))


def reset_edits(page_to_reset: str) -> None:
    state.edits[page_to_reset].clear()
    state.grid_version += 1


def clear_selections(page_to_clear: str) -> None:
    state.selections[page_to_clear]["locks"].clear()
    state.selections[page_to_clear]["excludes"].clear()
    state.grid_version += 1


def select_entries(entry_ids: list[str], selected: bool) -> None:
    """Button callback: select (or skip) every entry in `entry_ids`."""
    if selected:
        state.skipped_entries.difference_update(entry_ids)
    else:
        state.skipped_entries.update(entry_ids)
    state.entries_version += 1


def clear_fills(fill_format: str) -> None:
    """Button callback: drop every pick."""
    state.fills[fill_format].clear()
    state.fills_version += 1


def tick_entries(entry_ids: list[str], ticked: bool) -> None:
    """Assign dialog button callback: tick (or untick) every entry in `entry_ids`."""
    if ticked:
        state.assign_ticks.update(entry_ids)
    else:
        state.assign_ticks.difference_update(entry_ids)
    state.assign_version += 1


def save_fill_upload(text: str, file_name: str) -> None:
    """Download callback under Export to CSV: save the same file to the export folder."""
    try:
        path = gui.write_fill_upload(text, file_name)
    except OSError as exc:
        state.fill_notices.append(f"Export failed: {exc}. The download still has the file.")
        return
    state.fill_notices.append(f"Exported to {path}")


# --- Format and files ---

st.title("NFL DFS Optimizer")
page = st.radio("Mode", gui.PAGES, horizontal=True, key="w_format")
fmt = gui.page_format(page)

entries_use = {
    CLASSIC: "Classic reads kickoffs from it to seat the latest game in FLEX; exports "
    "and lineup downloads take their upload rows from it.",
    SHOWDOWN: "Exports and lineup downloads take their upload rows from it.",
    LATE_SWAP: "The entries to late swap, and the player pool whose Game Info "
    "decides which games have started.",
}[page]

with st.sidebar:
    st.header("Files")
    if hosted:
        projections_path = upload_file(
            "Projections",
            gui.PROJECTIONS_UPLOAD,
            f"A {gui.PROJECTIONS_GLOBS[fmt]} file; keep \"Early Slate\" or \"Late Slate\" "
            "in its name to tag the slate.",
        )
        entries_path = upload_file(
            "DraftKings entries",
            gui.ENTRIES_UPLOAD,
            "The DKEntries CSV from DraftKings"
            + ("." if page == LATE_SWAP else " (optional).")
            + f" {entries_use}",
        )
    else:
        projections_path = pick_file(
            "Projections",
            "file",
            gui.projection_files(fmt),
            f"Newest {gui.PROJECTIONS_GLOBS[fmt]}",
            allow_none=False,
        )
        entries_path = pick_file(
            "DraftKings entries",
            "entries",
            gui.entries_files(),
            f"Newest {common.DK_ENTRIES_GLOB}",
            allow_none=page != LATE_SWAP,
            help_text=entries_use,
        )

    # --- Settings ---
    st.header("Settings")
    roster = classic.ROSTER_SIZE if fmt == CLASSIC else 6
    num_lineups = 1
    if page != LATE_SWAP:
        num_lineups = st.number_input("Lineups", 1, 500, key=widget("num_lineups", 1))
    min_uniques = st.number_input(
        "Min uniques",
        1,
        roster,
        key=widget("min_uniques", 1),
        help=(
            "Players that must differ between any two entries in the same contest; "
            "locked players count, so it is relaxed where they already overlap."
            if page == LATE_SWAP
            else "Players (Showdown: roster spots) that must differ between any two lineups."
        ),
    )
    stack, stack_rb, max_te, no_dst_opp = 0, False, None, False
    max_salary = SALARY_CAP
    if fmt == CLASSIC:
        stack = st.number_input(
            "Stack: WR/TE with QB (0 = off)", 0, 4, key=widget("stack", 0)
        )
        stack_rb = st.checkbox("Stack an RB with the QB", key=widget("stack_rb", False))
        te_choice = st.selectbox("Max TE", ["No limit", 1, 2, 3], key=widget("max_te", "No limit"))
        max_te = None if te_choice == "No limit" else int(te_choice)
        no_dst_opp = st.checkbox("No DST vs. opposing offense", key=widget("no_dst_opp", False))
    min_salary = st.number_input(
        "Min salary (0 = no floor)",
        0,
        SALARY_CAP,
        step=100,
        key=widget("min_salary", 0),
        help=(
            "Covers the whole entry, locked players included; an entry that can't reach "
            "it is rebuilt without it."
            if page == LATE_SWAP
            else None
        ),
    )
    if fmt == SHOWDOWN:
        max_salary = st.number_input(
            "Max salary", 0, SALARY_CAP, step=100, key=widget("max_salary", SALARY_CAP)
        )
    target_label = st.radio(
        "Optimize on", list(gui.TARGET_CHOICES), horizontal=True, key=widget("target", "Projection")
    )
    ownership_field = classic.OWNERSHIP_LARGE_FIELD
    if fmt == CLASSIC:
        ownership_label = st.radio(
            "Ownership shown", list(gui.OWNERSHIP_CHOICES), horizontal=True,
            key=widget("ownership", "Large field"),
        )
        ownership_field = gui.OWNERSHIP_CHOICES[ownership_label]
    # Hosted: no export toggle; nothing is written outside the upload folder.
    export = False
    if not hosted:
        export = st.toggle(
            "Export to CSV",
            key=widget("export", False),
            help=(
                "Also save the upload-ready file on each run. " if page == LATE_SWAP else ""
            )
            + f"Writes to `{common.EXPORT_DIR}`",
        )

    settings = gui.Settings(
        num_lineups=int(num_lineups),
        min_uniques=int(min_uniques),
        stack=int(stack),
        stack_rb=stack_rb,
        max_te=max_te,
        no_dst_opp=no_dst_opp,
        min_salary=int(min_salary),
        max_salary=int(max_salary),
        target=gui.TARGET_CHOICES[target_label],
        ownership_field=ownership_field,
        export=export,
        dk_entries=entries_path,
    )
    errors = gui.validation_errors(page, settings)
    for error in errors:
        st.error(error)

# --- Load ---

if not projections_path and hosted:
    st.info(f"Upload a {gui.PROJECTIONS_GLOBS[fmt]} file in the sidebar.")
    st.stop()
if not projections_path:
    st.info(
        f"No {gui.PROJECTIONS_GLOBS[fmt]} file in {common.DOWNLOADS_DIR}. "
        f"Pick \"Other file...\" in the sidebar to use one elsewhere."
    )
    st.stop()
try:
    # Read once per run: the cache and the grid key must agree on which
    # version of the file this run shows, and a file deleted since the
    # picker listed it lands in the error below rather than a traceback.
    modified = os.path.getmtime(projections_path)
    pool = cached_pool(fmt, projections_path, modified)
except PlayerDataError as exc:
    st.error(f"Couldn't load {os.path.basename(projections_path)}: {exc}")
    st.stop()
except (ValueError, OSError) as exc:
    st.error(f"Couldn't load {projections_path}: {exc}")
    st.stop()
df = pool.df

slate = f" · {classic.detect_slate(projections_path)} Slate" if fmt == CLASSIC else ""
st.caption(f"{gui.describe_file(projections_path)}{slate} · {len(df)} players")
with st.expander("Load notes"):
    for note in gui.load_notes(fmt, pool, ownership_field):
        st.text(note)

# --- Player grid ---

selections = state.selections[page]
locks, excludes = selections["locks"], selections["excludes"]
edits = state.edits[page]
if gui.prune_edits(fmt, df, edits):
    state.grid_version += 1


def filter_key(name: str, options: list[str]) -> str:
    """A multiselect key whose remembered values are trimmed to this file's options."""
    key = widget(name, [])
    state[key] = [value for value in state[key] if value in options]
    return key


def clear_filters(page_to_clear: str) -> None:
    """Button callback: Position, Team and Search back to all players."""
    state[f"w_{page_to_clear}_positions"] = []
    state[f"w_{page_to_clear}_teams"] = []
    state[f"w_{page_to_clear}_search"] = ""


filter_cols = st.columns([2, 2, 3, 1], vertical_alignment="bottom")
position_options = gui.position_options(fmt, df["Position"].astype(str).unique())
team_options = sorted(df["Team"].astype(str).unique())
positions = filter_cols[0].multiselect(
    "Position", position_options, key=filter_key("positions", position_options)
)
teams = filter_cols[1].multiselect("Team", team_options, key=filter_key("teams", team_options))
search = filter_cols[2].text_input(
    "Search players", key=widget("search", ""), live=True, placeholder="Type a name"
)
filter_cols[3].button(
    "Clear filters",
    on_click=clear_filters,
    args=(page,),
    disabled=not (positions or teams or search),
    width="stretch",
)

grid_columns = gui.grid_columns(fmt)
view_cols = st.columns([2, 1, 1, 3], vertical_alignment="bottom")
sort_options = gui.sort_columns(grid_columns)
sort_key = widget("sort_by", gui.DEFAULT_SORT)
if state[sort_key] not in sort_options:
    state[sort_key] = gui.DEFAULT_SORT
sort_by = view_cols[0].selectbox(
    "Sort by",
    sort_options,
    key=sort_key,
    help="Kept while you filter or search. Clicking a column header also sorts, until "
    "the next filter change.",
)
descending = (
    view_cols[1].selectbox("Order", [DESCENDING, ASCENDING], key=widget("sort_order", DESCENDING))
    == DESCENDING
)
hideable = gui.hideable_columns(grid_columns)
show_keys = {column: widget(f"show_{column}", True) for column in hideable}


def show_all_columns(keys: list[str]) -> None:
    """Button callback: tick every column in the Columns list."""
    for key in keys:
        state[key] = True


hidden = sum(not state[key] for key in show_keys.values())
with view_cols[2].popover(
    f"Columns ({hidden} hidden)" if hidden else "Columns", width="stretch", key=f"columns_{page}"
):
    for column, key in show_keys.items():
        st.checkbox(column, key=key)
    st.button(
        "Show all",
        key=f"show_all_{page}",
        on_click=show_all_columns,
        args=(list(show_keys.values()),),
        disabled=not hidden,
    )
shown_columns = [column for column, key in show_keys.items() if state[key]]


def shown_frame():
    frame = gui.grid_frame(fmt, df, locks, excludes, edits)
    frame = gui.filter_frame(frame, positions, teams, search)
    return gui.sort_frame(frame, sort_by, descending)


shown = shown_frame()
grid_key = gui.grid_key(
    state.grid_version,
    page,
    projections_path,
    modified,
    positions,
    teams,
    search,
    (sort_by, descending),
)
if grid_key in state:
    outcome = gui.apply_grid_edits(
        fmt, shown, state[grid_key]["edited_rows"], df, locks, excludes, edits
    )
    state.notices.extend(outcome.notices)
    if outcome.reset_grid:
        state.grid_version += 1
        st.rerun()
    if outcome.changed:
        resorted = shown_frame()
        if gui.rows_moved(shown, resorted):
            # An edit to the sorted column moved players between rows, and the
            # pending edits are row positions: start a fresh editor.
            state.grid_version += 1
            st.rerun()
        shown = resorted

for notice in state.notices:
    st.warning(notice)
state.notices = []

st.caption(
    "Projection, Ceiling and ownership are editable: edits stay in this session and never "
    "change the file. Clear a cell, or type the file's number, to undo one edit. Edited "
    "players are highlighted and their changes listed in the Edited column."
    + (
        " **Showdown:** editing a FLEX Projection or Ceiling sets that player's Captain value "
        "to 1.5x the new number, even when the file supplied its own CPT value."
        if fmt == SHOWDOWN
        else ""
    )
)

# Grid cells are centered, except the Player and Edited text (and Showdown's
# Lock/Exclude, below). Headers can't be: Streamlit's grid always left-aligns them.
CENTER = "center"
SALARY_FORMAT = "$%,d"
OWN_FORMAT = "%.2f%%"


def own_column(label: str):
    """An editable ownership column: 0-100, shown as a percentage."""
    return st.column_config.NumberColumn(
        label, format=OWN_FORMAT, min_value=0.0, max_value=100.0, alignment=CENTER
    )


# The lineup and late-swap result tables: centered like the grid except Player.
# A config for a column a table lacks (Kickoff, Status) is ignored.
RESULT_COLUMNS = {
    **{
        column: st.column_config.Column(alignment=CENTER)
        for column in ("Slot", "Pos", "Team", "Kickoff (ET)", "Status")
    },
    "Salary": st.column_config.NumberColumn(format=SALARY_FORMAT, alignment=CENTER),
    "Proj": st.column_config.NumberColumn(format="%.2f", alignment=CENTER),
    "Own%": st.column_config.NumberColumn("Own", format=OWN_FORMAT, alignment=CENTER),
    "Ceiling": st.column_config.NumberColumn(format="%.2f", alignment=CENTER),
}

if fmt == CLASSIC:
    select_columns = {
        LOCK: st.column_config.CheckboxColumn(LOCK, alignment=CENTER),
        EXCLUDE: st.column_config.CheckboxColumn(EXCLUDE, alignment=CENTER),
    }
    own_columns = {
        "Small Field Own": own_column("Small Field Own"),
        "Large Field Own": own_column("Large Field Own"),
    }
else:
    # Selectbox cells have no alignment option; Streamlit draws them left-aligned.
    select_columns = {
        LOCK: st.column_config.SelectboxColumn(LOCK, options=list(gui.SHOWDOWN_SLOTS)),
        EXCLUDE: st.column_config.SelectboxColumn(EXCLUDE, options=list(gui.SHOWDOWN_SLOTS)),
    }
    own_columns = {
        "Own": own_column("Own"),
        "CPT Own": own_column("CPT Own"),
        "CPT Salary": st.column_config.NumberColumn(
            "CPT Salary", format=SALARY_FORMAT, alignment=CENTER
        ),
        "CPT Projection": st.column_config.NumberColumn(
            "CPT Proj", format="%.2f", alignment=CENTER
        ),
        "CPT Ceiling": st.column_config.NumberColumn(
            "CPT Ceiling", format="%.2f", alignment=CENTER
        ),
    }
editable = (LOCK, EXCLUDE, *gui.EDITABLE_COLUMNS[fmt])
st.data_editor(
    shown.style.apply(lambda _: gui.grid_styles(fmt, shown), axis=None),
    key=grid_key,
    hide_index=True,
    height=440,
    column_order=gui.column_order(shown, shown_columns),
    disabled=[c for c in shown.columns if c not in editable],
    column_config={
        **select_columns,
        "Position": st.column_config.TextColumn("Position", alignment=CENTER),
        "Team": st.column_config.TextColumn("Team", alignment=CENTER),
        "Opp": st.column_config.TextColumn("Opp", alignment=CENTER),
        "Salary": st.column_config.NumberColumn(
            "Salary", format=SALARY_FORMAT, alignment=CENTER
        ),
        "Projection": st.column_config.NumberColumn(
            "Projection", format="%.2f", min_value=0.0, alignment=CENTER
        ),
        "Ceiling": st.column_config.NumberColumn(
            "Ceiling", format="%.2f", min_value=0.0, alignment=CENTER
        ),
        **own_columns,
        gui.EDITED: st.column_config.TextColumn(gui.EDITED, width="medium"),
    },
)

summary_cols = st.columns([5, 1])
with summary_cols[0]:
    locked = gui.describe_selections(fmt, df, locks)
    excluded = gui.describe_selections(fmt, df, excludes)
    st.markdown(f"**Locked ({len(locked)}):** {', '.join(locked) or 'none'}")
    st.markdown(f"**Excluded ({len(excluded)}):** {', '.join(excluded) or 'none'}")
    missing = gui.missing_selections(fmt, df, locks) + gui.missing_selections(fmt, df, excludes)
    if missing:
        st.caption(f"{missing} selection(s) name players not in this file and are ignored.")
summary_cols[1].button(
    "Clear all locks/excludes",
    on_click=clear_selections,
    args=(page,),
    disabled=not (locks or excludes),
)

edit_cols = st.columns([5, 1])
with edit_cols[0]:
    edit_lines = gui.describe_edits(fmt, df, edits)
    st.markdown(f"**Edited ({len(edit_lines)}):** {'none' if not edit_lines else ''}")
    for line in edit_lines:
        st.markdown(f"- {line}")
    stale = len(set(edits) - gui.edited_keys(fmt, df, edits))
    if stale:
        st.caption(f"{stale} edit(s) name players not in this file and are ignored.")
edit_cols[1].button(
    "Reset to file values",
    on_click=reset_edits,
    args=(page,),
    disabled=not edits,
)

# --- Late swap ---


def draw_late_swap() -> None:
    """The Late swap page below the player grid: pick entries, run, show each entry."""
    st.divider()
    st.subheader("Entries to late swap")
    if not entries_path or not os.path.isfile(entries_path):
        st.info(
            "Pick a DraftKings entries file in the sidebar: download it from the Edit "
            "Entries page on DraftKings."
        )
        return
    skipped = state.skipped_entries
    try:
        entries_modified = os.path.getmtime(entries_path)
        entries = gui.entries_frame(entries_path, skipped)
    except (ValueError, OSError) as exc:
        st.error(f"Couldn't read {os.path.basename(entries_path)}: {exc}")
        return

    contest_names = gui.contest_labels(entries)
    contest_options = list(contest_names)
    contests = st.multiselect(
        "Contest",
        contest_options,
        key=filter_key("contests", contest_options),
        format_func=contest_names.get,
    )
    shown_entries = gui.filter_entries(entries, contests)
    entries_key = gui.entries_grid_key(
        state.entries_version, entries_path, entries_modified, contests
    )
    if entries_key in state and gui.apply_entry_edits(
        shown_entries, state[entries_key]["edited_rows"], skipped
    ):
        entries[gui.SWAP] = ~entries.index.isin(skipped)
        shown_entries = gui.filter_entries(entries, contests)

    st.data_editor(
        shown_entries,
        key=entries_key,
        hide_index=True,
        disabled=[c for c in shown_entries.columns if c != gui.SWAP],
        column_config={
            gui.SWAP: st.column_config.CheckboxColumn(gui.SWAP, width="small"),
            "Locked": st.column_config.TextColumn(
                "Locked", width="small", help="Slots whose games have started, as of now."
            ),
            "Lineup": st.column_config.TextColumn(
                "Lineup", width=gui.lineup_column_width(entries)
            ),
        },
    )
    chosen = [entry_id for entry_id in entries.index if entry_id not in skipped]
    shown_ids = list(shown_entries.index)
    pick_cols = st.columns([4, 1, 1])
    pick_cols[0].caption(
        f"{len(chosen)} of {len(entries)} entries selected. Entries you skip are left out "
        f"of the upload file, but still count toward Min uniques within their contest."
    )
    pick_cols[1].button(
        "Select all shown",
        on_click=select_entries,
        args=(shown_ids, True),
        disabled=all(i not in skipped for i in shown_ids),
    )
    pick_cols[2].button(
        "Skip all shown",
        on_click=select_entries,
        args=(shown_ids, False),
        disabled=all(i in skipped for i in shown_ids),
    )

    if st.button("Run late swap", type="primary", disabled=bool(errors) or not chosen):
        entry_ids = None if len(chosen) == len(entries) else frozenset(chosen)
        with st.spinner("Late swapping..."):
            state.results[LATE_SWAP] = gui.run_late_swap(
                df, projections_path, settings, locks, excludes, edits, entry_ids
            )

    run = state.results.get(LATE_SWAP)
    if run is None:
        return
    st.divider()
    if run.error:
        st.error(run.error)
    if run.export_error:
        st.warning(run.export_error)
    if run.notes:
        with st.expander("Run notes"):
            for note in run.notes:
                st.text(note)
    result = run.result
    if result is None:
        return
    st.success(result.summary)
    download_cols = st.columns([1, 4], vertical_alignment="center")
    download_cols[0].download_button(
        "Download filled entries CSV",
        run.upload_csv,
        file_name=run.file_name,
        mime="text/csv",
        key="download_late_swap",
        on_click="ignore",
    )
    download_cols[1].caption(
        (f"Exported to {run.export_path}. " if run.export_path else "")
        + f"Clock: {result.slate.now:%m/%d/%Y %I:%M%p} ET. "
        "From the last Run click; later changes aren't reflected until you run again."
        + (f" {gui.EDITED_MARK.strip()} = run with edited values." if run.edited else "")
    )

    status_options = list(gui.SWAP_STATUS_LABELS.values())
    show = st.multiselect("Show entries", status_options, key=widget("show_status", status_options))
    listed = [
        entry for entry in result.outcomes if gui.SWAP_STATUS_LABELS[entry.status] in show
    ]
    total = len(result.outcomes)
    for start in range(0, len(listed), 2):
        for column, entry in zip(st.columns(2), listed[start : start + 2]):
            with column:
                st.markdown(
                    f"**Entry {entry.number}/{total}: {entry.entry.entry_id}** · "
                    f"{entry.entry.contest_name} · *{gui.SWAP_STATUS_LABELS[entry.status]}*"
                )
                st.caption(gui.swap_totals(result, entry))
                if entry.note:
                    st.caption(f"Note: {entry.note}")
                st.dataframe(
                    gui.swap_table(result, entry, run.edited),
                    hide_index=True,
                    column_config=RESULT_COLUMNS,
                )


if page == LATE_SWAP:
    draw_late_swap()
    st.stop()

# --- Optimize ---

# Sent to the sidebar from here, after the grid, so a click sees this run's
# locks and edits; Streamlit still draws it at the sidebar's bottom.
if st.sidebar.button("Optimize", type="primary", disabled=bool(errors)):
    # The spinner too, so the feedback shows where the click was, not below the grid.
    with st.sidebar, st.spinner("Optimizing..."):
        state.results[fmt] = gui.optimize(
            fmt, df, projections_path, settings, locks, excludes, edits
        )
        # Lineup numbers now name new lineups; earlier picks keep their run.
        new_run = state.results[fmt]
        if new_run.result is not None and new_run.result.lineups:
            state.fill_runs[fmt] += 1
        state.fills_version += 1
        if hosted:
            # Built now, from this run's entries file: a later upload can't change it.
            run = state.results[fmt]
            state.all_lineups[fmt] = (
                gui.all_lineups_file_name(run),
                gui.all_lineups_csv(run, settings.dk_entries),
            )

# --- Fill entries ---


def fill_context() -> tuple:
    """
    (entries file, this run's assignable lineups by label, problem): the
    sidebar entries file read for this page's slots, or a problem
    ("info"/"error", text) when there is none or it can't be read.
    """
    if not entries_path or not os.path.isfile(entries_path):
        return None, {}, (
            "info",
            "Pick a DraftKings entries file in the sidebar to fill its entries with lineups.",
        )
    try:
        entries_modified = os.path.getmtime(entries_path)
        found = cached_fill_entries(fmt, entries_path, entries_modified)
    except (ValueError, OSError) as exc:
        return None, {}, ("error", f"Couldn't read {os.path.basename(entries_path)}: {exc}")
    lookup = cached_name_ids(entries_path, entries_modified)
    return found, gui.run_fills(run, state.fill_runs[fmt], lookup), None


def draw_fill_entries(found, current: dict, problem) -> None:
    """Above the lineups: the picks so far, and the upload file's download."""
    fills = state.fills[fmt]
    run_number = state.fill_runs[fmt]
    has_lineups = run is not None and run.result is not None and bool(run.result.lineups)
    if not has_lineups and not fills:
        return
    if run is None:  # otherwise the run's own divider sits just above
        st.divider()
    st.subheader("Fill entries")
    if problem:
        (st.info if problem[0] == "info" else st.error)(problem[1])
        return
    if has_lineups:
        unusable = [lu.number for lu in run.result.lineups if f"#{lu.number}" not in current]
        if unusable:
            st.warning(
                f"Lineup(s) {', '.join(f'#{n}' for n in unusable)} can't fill entries: "
                f"{os.path.basename(entries_path)} doesn't list all their players."
            )

    # Remove ticks take effect at once; the new version draws a fresh table.
    table_key = f"assigned_{state.fills_version}"
    if table_key in state and gui.apply_removals(
        gui.assignments_frame(found, fills, run_number), state[table_key]["edited_rows"], fills
    ):
        state.fills_version += 1
        table_key = f"assigned_{state.fills_version}"
    for notice in state.fill_notices:
        st.info(notice)
    state.fill_notices = []

    frame = gui.assignments_frame(found, fills, run_number)
    elsewhere = len(fills) - len(frame)
    caption = f"{len(frame)} of {len(found.entries)} entries have a lineup."
    if not fills:
        caption += " Use a lineup's Assign to entries button to fill some."
    if elsewhere:
        caption += f" {elsewhere} pick(s) for entries not in this file are left out."
    text = gui.fill_upload_csv(found, fills)
    file_name = gui.fill_file_name(fmt, projections_path)
    download_cols = st.columns([2, 1, 5], vertical_alignment="center")
    download_cols[0].download_button(
        "Download filled entries CSV",
        text,
        file_name=file_name,
        mime="text/csv",
        key=f"download_fill_{fmt}",
        disabled=frame.empty,
        on_click=save_fill_upload if export else "ignore",
        args=(text, file_name) if export else None,
    )
    # Clear all also drops picks for entries of another file.
    download_cols[1].button(
        "Clear all",
        key=f"clear_fills_{fmt}",
        on_click=clear_fills,
        args=(fmt,),
        disabled=not fills,
    )
    download_cols[2].caption(
        caption + (f" Downloading also saves it to `{common.EXPORT_DIR}`." if export else "")
    )
    if not frame.empty:
        with st.expander(f"Assignments ({len(frame)})"):
            st.data_editor(
                frame,
                key=table_key,
                hide_index=True,
                disabled=[c for c in frame.columns if c != gui.REMOVE],
                column_config={
                    gui.REMOVE: st.column_config.CheckboxColumn(
                        gui.REMOVE, help="Tick to take the lineup off this entry."
                    ),
                    gui.LINEUP: st.column_config.TextColumn(
                        gui.LINEUP, help="(run N) marks a lineup from an earlier Optimize."
                    ),
                    "Players": st.column_config.TextColumn(
                        "Players", width=gui.lineup_column_width(frame, "Players")
                    ),
                },
            )


@st.dialog("Assign to entries", width="large")
def assign_dialog(fill, found) -> None:
    """Tick the entries `fill` should fill; Save applies the ticks and closes."""
    fills = state.fills[fmt]
    run_number = state.fill_runs[fmt]
    ticked = state.assign_ticks
    label = fill.label(run_number)
    st.markdown(f"**Lineup {label}:** {fill.players}")
    frame = gui.assign_frame(found, fills, ticked, run_number)
    contest_names = gui.contest_labels(frame)
    contest_options = list(contest_names)
    contests = st.multiselect(
        "Contest",
        contest_options,
        key=filter_key("assign_contests", contest_options),
        format_func=contest_names.get,
    )
    shown = gui.filter_entries(frame, contests)
    grid_key = f"assign_{state.assign_version}_{'-'.join(contests)}"
    if grid_key in state and gui.apply_assign_edits(
        shown, state[grid_key]["edited_rows"], ticked
    ):
        frame = gui.assign_frame(found, fills, ticked, run_number)
        shown = gui.filter_entries(frame, contests)
    shown_ids = list(shown.index)
    button_cols = st.columns(2)
    button_cols[0].button(
        "Select all shown", on_click=tick_entries, args=(shown_ids, True), width="stretch"
    )
    button_cols[1].button(
        "Clear shown", on_click=tick_entries, args=(shown_ids, False), width="stretch"
    )
    st.data_editor(
        shown,
        key=grid_key,
        hide_index=True,
        disabled=[c for c in shown.columns if c != gui.ASSIGN],
        column_config={
            gui.ASSIGN: st.column_config.CheckboxColumn(gui.ASSIGN),
            gui.ASSIGNED_TO: st.column_config.TextColumn(
                gui.ASSIGNED_TO,
                help="The lineup the entry holds now; ticking it replaces that lineup.",
            ),
        },
    )
    replacing = sum(1 for entry_id in ticked if entry_id in fills and fills[entry_id] != fill)
    st.caption(
        f"{len(ticked)} entr{'y' if len(ticked) == 1 else 'ies'} ticked"
        + (f"; {replacing} will replace another lineup." if replacing else ".")
    )
    action_cols = st.columns(2)
    if action_cols[0].button("Save", type="primary", width="stretch"):
        counts = gui.save_assignment(fill, ticked, fills)
        state.fill_notices.append(gui.assignment_notice(label, *counts))
        state.fills_version += 1
        st.rerun()
    if action_cols[1].button("Cancel", width="stretch"):
        st.rerun()


run = state.results.get(fmt)
if run is not None:
    st.divider()
    if run.error:
        st.error(run.error)
    else:
        result = run.result
        if result.warning:
            st.warning(result.warning.strip())
        if result.messages:
            text = "\n\n".join(m.strip() for m in result.messages)
            (st.error if not result.lineups else st.warning)(text)
        if run.export_path:
            st.success(f"Exported to {run.export_path}")
    if run.notes:
        with st.expander("Run notes"):
            for note in run.notes:
                st.text(note)

# Above the lineups, so a long run doesn't push it out of sight.
fill_found, fill_current, fill_problem = fill_context()
draw_fill_entries(fill_found, fill_current, fill_problem)

if run is not None and run.result is not None and run.result.lineups:
    lineups = run.result.lineups
    st.divider()
    st.subheader(f"{len(lineups)} lineup{'s' if len(lineups) != 1 else ''}")
    st.caption(
        "From the last Optimize click; later changes aren't reflected until you run again."
        + (f" {gui.EDITED_MARK.strip()} = run with edited values." if run.edited else "")
    )
    if hosted and fmt in state.all_lineups:
        all_name, all_csv = state.all_lineups[fmt]
        st.download_button(
            "Download all lineups",
            all_csv,
            file_name=all_name,
            mime="text/csv",
            key=f"download_all_{run.fmt}",
            help="Every lineup in one file: what -e would export for this run.",
            on_click="ignore",
        )
    show_kickoff = getattr(run.result, "show_kickoff", False)
    heading = f"{run.slate} Slate Lineup" if run.fmt == CLASSIC else "Showdown Lineup"
    assigning = None
    for start in range(0, len(lineups), 2):
        pairs = zip(lineups[start : start + 2], run.lineup_csvs[start : start + 2])
        for column, (lineup, lineup_csv) in zip(st.columns(2), pairs):
            with column:
                st.markdown(f"**{heading} #{lineup.number}**")
                st.caption(gui.lineup_totals(run.fmt, lineup, run.target))
                fill = fill_current.get(f"#{lineup.number}")
                if fill is not None:
                    st.caption(gui.assignment_summary(fill, state.fills[fmt], fill_found))
                st.dataframe(
                    gui.lineup_table(run.fmt, lineup, show_kickoff, run.edited),
                    hide_index=True,
                    column_config=RESULT_COLUMNS,
                )
                card_cols = st.columns(2)
                card_cols[0].download_button(
                    "Download CSV",
                    lineup_csv,
                    file_name=gui.lineup_file_name(run, lineup.number),
                    mime="text/csv",
                    key=f"download_{run.fmt}_{lineup.number}",
                    help="The -e export rows for this lineup, with the DKEntries upload row last.",
                    on_click="ignore",
                )
                if fill is not None:
                    assign_help = "Pick the DraftKings entries this lineup fills."
                elif fill_problem:
                    assign_help = fill_problem[1]
                else:
                    assign_help = "The entries file doesn't list all of this lineup's players."
                if card_cols[1].button(
                    "Assign to entries",
                    key=f"assign_{run.fmt}_{lineup.number}",
                    disabled=fill is None,
                    help=assign_help,
                ):
                    assigning = fill
    # Opened here, outside the columns; the ticks start from the entries the
    # lineup already fills.
    if assigning is not None:
        state.assign_ticks = gui.holders(assigning, state.fills[fmt])
        state.assign_version += 1
        assign_dialog(assigning, fill_found)

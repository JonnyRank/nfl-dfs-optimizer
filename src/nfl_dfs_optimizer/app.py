"""
Streamlit app for the Classic and Showdown optimizers.

Run it with the launcher ("NFL DFS Optimizer.bat" in the repo root) or:

    uv run streamlit run src/nfl_dfs_optimizer/app.py

The logic lives in gui.py; this file only draws widgets and keeps state.
Widget keys start with "w_" and are re-assigned at the top of every run, so a
value survives while its widget is hidden (the other format's settings). Locks
and excludes live in st.session_state["selections"], keyed by player, not in
the grid, so they survive filtering, sorting, and a re-downloaded file.
"""

import hashlib
import os

import streamlit as st

from nfl_dfs_optimizer import classic, common, gui
from nfl_dfs_optimizer.common import SALARY_CAP, PlayerDataError
from nfl_dfs_optimizer.gui import CLASSIC, EXCLUDE, LOCK, SHOWDOWN

NEWEST = "__newest__"
OTHER = "__other__"
NONE = "__none__"

st.set_page_config(page_title="NFL DFS Optimizer", layout="wide")

state = st.session_state
# Streamlit drops a widget's value on any run that doesn't draw it; assigning
# it back keeps the hidden format's settings.
for widget_key in list(state.keys()):
    if isinstance(widget_key, str) and widget_key.startswith("w_"):
        state[widget_key] = state[widget_key]
state.setdefault("selections", {f: {"locks": {}, "excludes": {}} for f in gui.FORMATS})
state.setdefault("grid_version", 0)
state.setdefault("results", {})
state.setdefault("notices", [])


@st.cache_data(show_spinner="Loading projections...")
def cached_pool(fmt: str, path: str, modified: float):
    """The loaded file; `modified` makes a re-saved file load again."""
    return gui.load_pool(fmt, path)


def widget(name: str, default):
    """The session key for this format's `name` widget, seeded with `default`."""
    key = f"w_{fmt}_{name}"
    state.setdefault(key, default)
    return key


def pick_file(label: str, name: str, files: list[str], newest_label: str, allow_none: bool):
    """A file chooser: newest match (default), a specific match, none, or a typed path."""
    options = [NEWEST, *([NONE] if allow_none else []), *files, OTHER]
    key = widget(name, NEWEST)
    if state[key] not in options:
        state[key] = NEWEST
    labels = {NEWEST: newest_label, NONE: "None", OTHER: "Other file..."}
    choice = st.selectbox(
        label, options, key=key, format_func=lambda o: labels.get(o, os.path.basename(o))
    )
    if choice == NEWEST:
        path = files[0] if files else None
    elif choice == NONE:
        path = None
    elif choice == OTHER:
        typed = st.text_input(f"{label} path", key=widget(f"{name}_other", ""))
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


def clear_selections(fmt_to_clear: str) -> None:
    state.selections[fmt_to_clear]["locks"].clear()
    state.selections[fmt_to_clear]["excludes"].clear()
    state.grid_version += 1


# --- Format and files ---

st.title("NFL DFS Optimizer")
fmt = st.radio("Format", gui.FORMATS, horizontal=True, key="w_format")

with st.sidebar:
    st.header("Files")
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
        allow_none=True,
    )
    st.caption(
        "Classic reads kickoffs from it to seat the latest game in FLEX; an export "
        "takes its upload row from it."
        if fmt == CLASSIC
        else "An export takes its upload row from it."
    )

    # --- Settings ---
    st.header("Settings")
    roster = classic.ROSTER_SIZE if fmt == CLASSIC else 6
    num_lineups = st.number_input("Lineups", 1, 500, key=widget("num_lineups", 1))
    min_uniques = st.number_input(
        "Min uniques",
        1,
        roster,
        key=widget("min_uniques", 1),
        help="Players (Showdown: roster spots) that must differ between any two lineups.",
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
        "Min salary (0 = no floor)", 0, SALARY_CAP, step=100, key=widget("min_salary", 0)
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
    export = st.toggle("Export to CSV", key=widget("export", False))
    if export:
        st.caption(f"Writes to {common.EXPORT_DIR}")

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
    errors = gui.validation_errors(fmt, settings)
    for error in errors:
        st.error(error)

# --- Load ---

if not projections_path:
    st.info(
        f"No {gui.PROJECTIONS_GLOBS[fmt]} file in {common.DOWNLOADS_DIR}. "
        f"Pick \"Other file...\" in the sidebar to use one elsewhere."
    )
    st.stop()
try:
    pool = cached_pool(fmt, projections_path, os.path.getmtime(projections_path))
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

selections = state.selections[fmt]
locks, excludes = selections["locks"], selections["excludes"]



def filter_key(name: str, options: list[str]) -> str:
    """A multiselect key whose remembered values are trimmed to this file's options."""
    key = widget(name, [])
    state[key] = [value for value in state[key] if value in options]
    return key


filter_cols = st.columns([2, 2, 3])
position_options = sorted(df["Position"].astype(str).unique())
team_options = sorted(df["Team"].astype(str).unique())
positions = filter_cols[0].multiselect(
    "Position", position_options, key=filter_key("positions", position_options)
)
teams = filter_cols[1].multiselect("Team", team_options, key=filter_key("teams", team_options))
search = filter_cols[2].text_input("Search players", key=widget("search", ""))


def shown_frame():
    return gui.filter_frame(gui.grid_frame(fmt, df, locks, excludes), positions, teams, search)


shown = shown_frame()
signature = hashlib.md5(
    repr((fmt, projections_path, positions, teams, search)).encode()
).hexdigest()[:10]
grid_key = f"grid_{state.grid_version}_{signature}"
if grid_key in state:
    outcome = gui.apply_grid_edits(
        fmt, shown, state[grid_key]["edited_rows"], df, locks, excludes
    )
    state.notices.extend(outcome.notices)
    if outcome.reset_grid:
        state.grid_version += 1
        st.rerun()
    if outcome.changed:
        shown = shown_frame()

for notice in state.notices:
    st.warning(notice)
state.notices = []

if fmt == CLASSIC:
    select_columns = {
        LOCK: st.column_config.CheckboxColumn(LOCK, width="small"),
        EXCLUDE: st.column_config.CheckboxColumn(EXCLUDE, width="small"),
    }
    own_columns = {
        "Small Field Own": st.column_config.NumberColumn("Small Field Own %", format="%.2f"),
        "Large Field Own": st.column_config.NumberColumn("Large Field Own %", format="%.2f"),
    }
else:
    select_columns = {
        LOCK: st.column_config.SelectboxColumn(LOCK, options=list(gui.SHOWDOWN_SLOTS), width="small"),
        EXCLUDE: st.column_config.SelectboxColumn(
            EXCLUDE, options=list(gui.SHOWDOWN_SLOTS), width="small"
        ),
    }
    own_columns = {
        "Own": st.column_config.NumberColumn("Own %", format="%.2f"),
        "CPT Own": st.column_config.NumberColumn("CPT Own %", format="%.2f"),
        "CPT Salary": st.column_config.NumberColumn("CPT Salary", format="$%d"),
        "CPT Projection": st.column_config.NumberColumn("CPT Proj", format="%.2f"),
        "CPT Ceiling": st.column_config.NumberColumn("CPT Ceiling", format="%.2f"),
    }
st.data_editor(
    shown,
    key=grid_key,
    hide_index=True,
    height=440,
    disabled=[c for c in shown.columns if c not in (LOCK, EXCLUDE)],
    column_config={
        **select_columns,
        "Salary": st.column_config.NumberColumn("Salary", format="$%d"),
        "Projection": st.column_config.NumberColumn("Projection", format="%.2f"),
        "Ceiling": st.column_config.NumberColumn("Ceiling", format="%.2f"),
        **own_columns,
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
    args=(fmt,),
    disabled=not (locks or excludes),
)

# --- Optimize ---

if st.button("Optimize", type="primary", disabled=bool(errors)):
    with st.spinner("Optimizing..."):
        state.results[fmt] = gui.optimize(
            fmt, df, projections_path, settings, locks, excludes
        )

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
    if run.result is not None and run.result.lineups:
        lineups = run.result.lineups
        st.subheader(f"{len(lineups)} lineup{'s' if len(lineups) != 1 else ''}")
        st.caption("From the last Optimize click; later changes aren't reflected until you run again.")
        show_kickoff = getattr(run.result, "show_kickoff", False)
        heading = f"{run.slate} Slate Lineup" if run.fmt == CLASSIC else "Showdown Lineup"
        for start in range(0, len(lineups), 2):
            for column, lineup in zip(st.columns(2), lineups[start : start + 2]):
                with column:
                    st.markdown(f"**{heading} #{lineup.number}**")
                    st.caption(gui.lineup_totals(run.fmt, lineup, run.target))
                    st.dataframe(
                        gui.lineup_table(run.fmt, lineup, show_kickoff),
                        hide_index=True,
                        column_config={
                            "Salary": st.column_config.NumberColumn(format="$%d"),
                            "Proj": st.column_config.NumberColumn(format="%.2f"),
                            "Own%": st.column_config.NumberColumn(format="%.2f"),
                            "Ceiling": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )

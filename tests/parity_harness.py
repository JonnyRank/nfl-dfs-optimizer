"""
Parity harness: runs the NFL optimizer CLIs over a matrix of flag combinations
and records what they produce, so a refactor can prove it changed nothing.

Each case runs a CLI's main() (nfl_dfs_optimizer.cli) in-process on the committed, obfuscated fixtures
in tests/fixtures/public/ (see tests/fixtures/make_public_fixtures.py), with
the export and Downloads folders pointed at temporary directories, and records
three goldens in tests/fixtures/golden/<format>/:

    <case>.json        lineup count and, per lineup, objective score and salary
    <case>.txt         the full stdout, with paths, file times and timestamps
                       replaced by placeholders
    <case>.export.csv  the exported CSV (-e), or the late-swap upload file (-ls)

The summary (.json) is the parity contract: solver ties may legitimately swap
equal-score players, so it compares scores and salaries, not players. The
transcript and export pin the CLI's printed and written output on top of that.

Regenerate the goldens (only when a behavior change is intended):

    uv run python tests/parity_harness.py --regen
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "tests", "fixtures", "public")
GOLDEN = os.path.join(ROOT, "tests", "fixtures", "golden")

CLASSIC_FILE = os.path.join(PUBLIC, "DraftKings NFL DFS Projections -- Main Slate.csv")
SHOWDOWN_FILE = os.path.join(PUBLIC, "DK NFL Showdown Projections.csv")
CLASSIC_ENTRIES = os.path.join(PUBLIC, "DKEntriesClassic.csv")
LATE_SWAP_ENTRIES = os.path.join(PUBLIC, "DKEntriesClassicLateSwap.csv")
SHOWDOWN_ENTRIES = os.path.join(PUBLIC, "DKEntriesShowdown.csv")

# Placeholders the transcript and export use in place of machine-specific text.
FIXTURES_TOKEN = "<FIXTURES>"
EXPORT_TOKEN = "<EXPORT_DIR>"
DOWNLOADS_TOKEN = "<DOWNLOADS>"
TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
MODIFIED_PATTERN = re.compile(r"modified \w{3} \d{2}/\d{2} \d{2}:\d{2}[AP]M")

# Objective weights by target, matching OPTIMIZATION_TARGETS in both scripts.
TARGET_WEIGHTS: dict[str, tuple[float, float]] = {
    "projection": (1.0, 0.0),
    "ceiling": (0.0, 1.0),
    "blend": (0.5, 0.5),
}


@dataclass(frozen=True)
class Case:
    """
    One CLI invocation. `entries` False runs with no DKEntries file at all.
    `now` (Eastern, "YYYY-MM-DD HH:MM") freezes the clock late swap reads.
    """

    fmt: str  # "classic" or "showdown"
    name: str
    args: tuple[str, ...]
    entries: bool = True
    now: str | None = None

    @property
    def target(self) -> str:
        if "-c" in self.args:
            return "ceiling"
        if "-pj" in self.args:
            return "blend"
        return "projection"

    @property
    def exports(self) -> bool:
        """True when the case writes a file: -e lineups, or the -ls upload file."""
        return "-ls" in self.args or "-e" in self.args

    def argv(self) -> list[str]:
        projections = CLASSIC_FILE if self.fmt == "classic" else SHOWDOWN_FILE
        entries = CLASSIC_ENTRIES if self.fmt == "classic" else SHOWDOWN_ENTRIES
        if self.now:
            entries = LATE_SWAP_ENTRIES
        extra = ["-dk", entries] if self.entries else []
        return [projections, *self.args, *extra]


def _c(name: str, *args: str, entries: bool = True, now: str | None = None) -> Case:
    return Case("classic", name, args, entries, now)


def _s(name: str, *args: str, entries: bool = True) -> Case:
    return Case("showdown", name, args, entries)


CASES: list[Case] = [
    # --- Classic ---
    _c("base", "-n", "1"),
    _c("multi_u2", "-n", "10", "-u", "2"),
    _c("multi_u4", "-n", "5", "-u", "4"),
    _c("lock", "-n", "3", "-u", "2", "-l", "Josh Allen", "Trey McBride"),
    _c("lock_missing", "-n", "2", "-l", "josh allen", "Nobody Here", "Seahawks"),
    _c("exclude", "-n", "3", "-u", "2", "-x", "Jahmyr Gibbs", "Christian McCaffrey"),
    _c("lock_exclude", "-n", "3", "-u", "2", "-l", "Lamar Jackson", "-x", "Derrick Henry"),
    _c("stack1", "-n", "3", "-u", "2", "-s"),
    _c("stack2", "-n", "3", "-u", "2", "-s", "2"),
    _c("stack_rb", "-n", "3", "-u", "2", "-srb"),
    _c("max_te1", "-n", "3", "-u", "2", "-te", "1"),
    _c("max_te2_lock_te", "-n", "3", "-te", "2", "-l", "Brock Bowers", "Trey McBride"),
    _c("no_dst_opp", "-n", "3", "-u", "2", "-ndo"),
    _c("min_salary", "-n", "5", "-u", "2", "-mns", "49900"),
    _c("ceiling", "-n", "3", "-u", "2", "-c"),
    _c("projceiling", "-n", "3", "-u", "2", "-pj"),
    _c("small_field", "-n", "3", "-u", "2", "-sf"),
    _c("export", "-n", "3", "-u", "2", "-e"),
    _c("export_ceiling_sf", "-n", "2", "-c", "-sf", "-e"),
    _c("export_no_entries", "-n", "2", "-u", "2", "-e", entries=False),
    _c("no_entries", "-n", "3", "-u", "2", entries=False),
    _c(
        "combo",
        "-n", "5", "-u", "2", "-s", "-srb", "-te", "1", "-ndo", "-mns", "49500",
        "-l", "Josh Allen", "-x", "Dalton Kincaid", "-pj", "-sf", "-e",
    ),
    _c("infeasible_two_qbs", "-n", "3", "-mns", "49000", "-l", "Josh Allen", "Lamar Jackson"),
    _c(
        "exhausted",
        "-n", "5", "-u", "3", "-mns", "49000",
        "-l", "Patrick Mahomes", "Jahmyr Gibbs", "Kenneth Walker III", "Parker Washington",
        "Rashod Bateman", "Malik Washington", "Mark Andrews",
    ),
    _c("min_salary_cap_stack", "-n", "2", "-mns", "50000", "-s", "3", "-srb"),
    _c("bad_min_salary", "-n", "2", "-mns", "60000"),
    # Late swap, on DKEntriesClassicLateSwap.csv with the clock frozen: before
    # any kickoff, after the 1:00PM games start, and after the 4:05PM ones.
    _c("late_swap_pregame", "-ls", "-u", "2", now="2026-09-27 11:00"),
    _c("late_swap_1pm", "-ls", "-u", "2", "-mns", "49500", "-s", now="2026-09-27 13:30"),
    _c(
        "late_swap_405",
        "-ls", "-u", "3", "-pj", "-sf", "-te", "1", "-ndo", "-n", "3", "-e",
        "-l", "Brock Purdy", "-x", "George Kittle",
        now="2026-09-27 16:10",
    ),
    # --- Showdown ---
    _s("base", "-n", "1"),
    _s("multi_u2", "-n", "10", "-u", "2"),
    _s("multi_u4", "-n", "5", "-u", "4"),
    _s("lock_cpt", "-n", "5", "-u", "2", "-l", "Jordan Love:CPT"),
    _s("lock_flex_any", "-n", "5", "-u", "2", "-l", "Bijan Robinson:FLEX", "Tucker Kraft"),
    _s("exclude_cpt", "-n", "5", "-u", "2", "-x", "Bijan Robinson:CPT"),
    _s("exclude_any_flex", "-n", "5", "-u", "2", "-x", "Jordan Love", "Drake London:FLEX"),
    _s(
        "lock_repeat_missing",
        "-n", "2", "-l", "Jordan Love", "jordan love", "Nobody Here", "Packers:cpt",
    ),
    _s("lock_exclude_slots", "-n", "4", "-u", "2", "-l", "Drake London:CPT", "-x", "Jordan Love:FLEX"),
    _s("max_salary", "-n", "5", "-u", "2", "-ms", "49000"),
    _s("max_salary_clamped", "-n", "2", "-ms", "60000"),
    _s("min_salary", "-n", "5", "-u", "2", "-mns", "49800"),
    _s("salary_window", "-n", "5", "-u", "2", "-ms", "49800", "-mns", "49500"),
    _s("ceiling", "-n", "5", "-u", "2", "-c"),
    _s("projceiling", "-n", "5", "-u", "3", "-pj"),
    _s("export", "-n", "3", "-u", "2", "-e"),
    _s("export_no_entries", "-n", "2", "-e", entries=False),
    _s("export_ceiling", "-n", "2", "-u", "2", "-c", "-e"),
    _s(
        "combo",
        "-n", "5", "-u", "3", "-l", "Jordan Love:CPT", "-x", "Bijan Robinson:FLEX",
        "-ms", "49900", "-mns", "48000", "-pj", "-e",
    ),
    _s("infeasible_two_cpt", "-n", "2", "-l", "Jordan Love:CPT", "Bijan Robinson:CPT"),
    _s("exhausted_u6", "-n", "10", "-u", "6", "-mns", "45000", "-l", "Jordan Love", "Bijan Robinson"),
    _s("bad_min_salary", "-n", "2", "-ms", "45000", "-mns", "46000"),
]

CASES_BY_ID: dict[str, Case] = {f"{case.fmt}/{case.name}": case for case in CASES}


# --- Running a CLI ---


def frozen_datetime(now: str) -> type[datetime]:
    """A datetime class whose now() is `now` Eastern, for the late-swap clock."""
    frozen = datetime.strptime(now, "%Y-%m-%d %H:%M").replace(
        tzinfo=ZoneInfo("America/New_York")
    )

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return frozen.astimezone(tz) if tz else frozen.replace(tzinfo=None)

    return FrozenDatetime


@contextlib.contextmanager
def patched_cli(fmt: str, export_dir: str, downloads_dir: str, now: str | None) -> Iterator[Callable[[], None]]:
    """
    Yields a format's CLI main() with the export and Downloads folders pointed
    at temporary directories and, for late swap, the clock frozen at `now`.

    The goldens were recorded from the legacy scripts before the package
    existed; since then the scripts in legacy/ are shims over these same CLIs.
    """
    from nfl_dfs_optimizer import common, late_swap
    from nfl_dfs_optimizer.cli import classic, showdown

    saved = (common.EXPORT_DIR, common.DOWNLOADS_DIR, late_swap.datetime)
    common.EXPORT_DIR, common.DOWNLOADS_DIR = export_dir, downloads_dir
    if now:
        late_swap.datetime = frozen_datetime(now)
    try:
        yield (classic if fmt == "classic" else showdown).main
    finally:
        common.EXPORT_DIR, common.DOWNLOADS_DIR, late_swap.datetime = saved


@dataclass
class CaseOutput:
    transcript: str
    export: str | None


def normalize(text: str, export_dir: str, downloads_dir: str) -> str:
    """Replaces paths, file times and timestamps with stable placeholders."""
    for path, token in (
        (export_dir, EXPORT_TOKEN),
        (downloads_dir, DOWNLOADS_TOKEN),
        (PUBLIC, FIXTURES_TOKEN),
    ):
        text = text.replace(path, token)
    text = TIMESTAMP_PATTERN.sub("<TIMESTAMP>", text)
    text = MODIFIED_PATTERN.sub("modified <MTIME>", text)
    # os.path.join writes "\" on Windows and "/" elsewhere; only placeholder
    # paths remain by now, so unify their separators.
    text = re.sub(r"(<[A-Z_]+>)\\", r"\1/", text)
    return text


def run_case(case: Case) -> CaseOutput:
    """Runs one case's CLI in-process and returns its normalized output."""
    with tempfile.TemporaryDirectory() as export_dir, tempfile.TemporaryDirectory() as downloads_dir:
        buffer = io.StringIO()
        saved_argv = sys.argv
        sys.argv = ["optimizer", *case.argv()]
        try:
            with (
                patched_cli(case.fmt, export_dir, downloads_dir, case.now) as main,
                contextlib.redirect_stdout(buffer),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                main()
        finally:
            sys.argv = saved_argv
        # Lineup exports land in the export folder; late swap's upload file
        # lands in Downloads. A run writes at most one of them.
        written = [
            os.path.join(folder, name)
            for folder in (export_dir, downloads_dir)
            for name in sorted(os.listdir(folder))
        ]
        assert len(written) <= 1, written
        export = None
        if written:
            with open(written[0], encoding="utf-8") as handle:
                export = f"# {os.path.basename(written[0])}\n" + handle.read()
            export = normalize(export, export_dir, downloads_dir)
        return CaseOutput(normalize(buffer.getvalue(), export_dir, downloads_dir), export)


# --- Summaries ---

_HEADER = re.compile(r"^--- (Optimal NFL .*Lineup #\d+|Entry \d+/\d+: .*) ---$")
_TOTAL = re.compile(r"^(Projection|Ceiling|Salary): \$?([\d,.-]+)")
# A late-swap entry's totals print as "before -> after"; the after value counts.
_SWAP_TOTAL = re.compile(r"(Projection|Ceiling): [\d.-]+ -> ([\d.-]+)|(Salary): \$([\d,]+)")


def summarize(case: Case, transcript: str) -> dict[str, object]:
    """Reads each printed lineup's totals and scores it under the case's target."""
    proj_weight, ceiling_weight = TARGET_WEIGHTS[case.target]
    lineups: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for line in transcript.splitlines():
        header = _HEADER.match(line)
        if header:
            current = {}
            lineups.append(current)
            continue
        if current is None:
            continue
        if " -> " in line:
            for match in _SWAP_TOTAL.finditer(line):
                key = match.group(1) or match.group(3)
                value = match.group(2) or match.group(4)
                current.setdefault(key, float(value.replace(",", "")))
            continue
        total = _TOTAL.match(line)
        if total and total.group(1) not in current:
            current[total.group(1)] = float(total.group(2).replace(",", ""))
    return {
        "count": len(lineups),
        "lineups": [
            {
                "score": round(proj_weight * lu["Projection"] + ceiling_weight * lu["Ceiling"], 2),
                "salary": int(lu["Salary"]),
            }
            for lu in lineups
        ],
    }


def golden_paths(case: Case) -> dict[str, str]:
    base = os.path.join(GOLDEN, case.fmt, case.name)
    return {"json": base + ".json", "txt": base + ".txt", "export": base + ".export.csv"}


def load_golden(case: Case) -> tuple[dict[str, object], str, str | None]:
    paths = golden_paths(case)
    with open(paths["json"], encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(paths["txt"], encoding="utf-8") as handle:
        transcript = handle.read()
    export = None
    if os.path.exists(paths["export"]):
        with open(paths["export"], encoding="utf-8") as handle:
            export = handle.read()
    return summary, transcript, export


def write_golden(case: Case) -> dict[str, object]:
    output = run_case(case)
    summary = {"args": list(case.args), "entries": case.entries, **summarize(case, output.transcript)}
    paths = golden_paths(case)
    os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
    with open(paths["json"], "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with open(paths["txt"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write(output.transcript)
    if output.export is not None:
        with open(paths["export"], "w", encoding="utf-8", newline="\n") as handle:
            handle.write(output.export)
    elif os.path.exists(paths["export"]):
        os.remove(paths["export"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--regen", action="store_true", help="Rewrite every golden.")
    parser.add_argument("cases", nargs="*", help="Case ids (classic/base ...); default all.")
    args = parser.parse_args()
    selected = [CASES_BY_ID[c] for c in args.cases] if args.cases else CASES
    for case in selected:
        if args.regen:
            summary = write_golden(case)
        else:
            summary = summarize(case, run_case(case).transcript)
        scores = ", ".join(f"{lu['score']:.2f}/${lu['salary']:,}" for lu in summary["lineups"])
        print(f"{case.fmt}/{case.name}: {summary['count']} lineup(s) {scores}")


if __name__ == "__main__":
    main()

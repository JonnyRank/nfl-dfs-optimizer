# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Persona & audience

From `project_rules.md`: act as a Principal Python Architect (data ingestion, analysis, optimization). Prioritize clean idiomatic Python, performance, and robust error handling. The user reads and understands code but does not write it — explain changes in terms of behavior and trade-offs, not line-by-line diffs.

## What this is

Standalone DraftKings DFS lineup optimizers. Each `.py` file at the repo root is a self-contained script — there is no package, no shared module, no test suite, and no build step. Common logic (data cleaning, diversity loop, table printing) is **duplicated across scripts by design**; a fix in one does not propagate, so when changing shared-looking behavior, decide explicitly whether the sibling scripts need the same change.

| Script | Sport / format | Roster | Solver | Input |
| --- | --- | --- | --- | --- |
| `NFL-Multi-Opto-v2.0.py` | NFL Classic, N lineups | 9 (QB/2RB/3WR/TE/FLEX/DST) | HiGHS (`pulp.HiGHS`, needs `highspy`) | newest `DraftKings NFL DFS Projections*.csv` in Downloads, or path as argv |
| `NFL-SD-Multi-Opto-v1.0.py` | NFL Showdown (Captain Mode), N lineups | 6 (1 CPT + 5 FLEX) | HiGHS (`pulp.HiGHS`, needs `highspy`) | newest `DK NFL Showdown Projections*.csv` in Downloads, or path as argv |
| `NBA-Multi-Opto-v1.0.py` | NBA, N lineups | 8 | CBC | auto-globbed files |
| `NBA-Single-Opto-v1.0.py` | NBA, 1 lineup | 8 | CBC | auto-globbed files |

The NFL scripts are the modern generation: module docstring, typed helpers, `argparse`, `load_player_data()` / `main()` structure, everything wrapped in one `try/except` in `main()`. The NBA scripts are the older generation: top-level procedural code that runs at import, module-level config constants (`TARGET_DIRECTORY`, `NUMBER_OF_LINEUPS`, `MIN_UNIQUES`), and input files discovered by glob (`DKEntries*.csv`, `NBA-Projs-*.csv`) instead of CLI args. Follow the NFL style for new work.

## Running

```bash
venv/Scripts/python.exe NFL-Multi-Opto-v2.0.py "C:\path\to\projections.csv" -n 5 -u 2 -e -l "Josh Allen" -s -ndo
venv/Scripts/python.exe NFL-SD-Multi-Opto-v1.0.py "C:\path\to\showdown.csv" -n 5 -u 2 -e -l "Drake Maye:CPT" -ms 49800
venv/Scripts/python.exe NBA-Multi-Opto-v1.0.py   # no args; edit the constants at the top of the file
```

Local env is a plain `venv/` (Python 3.13) plus `requirements.txt` (`pandas`, `pulp`, `highspy`, `tzdata` — the Eastern zone for late swap; Windows has no zone database). There is no lint or test command. Verification means running a script against a real projections CSV and reading the printed lineups.

Note: `.claude/hooks/session-start.sh` bootstraps with `uv sync` and a `.python-version` pin — neither `pyproject.toml` nor `.python-version` exists here, so that hook only matters if the repo is migrated to uv (it is gated on `CLAUDE_CODE_REMOTE=true` and no-ops locally).

### CLI flags (NFL scripts)

`filepath` (optional positional; `find_projections_file()` uses it as given — a file, never a folder — else the newest `PROJECTIONS_GLOB` match in `DOWNLOADS_DIR`, else raises), `-n/--num-lineups`, `-u/--min-uniques`, `-e/--export`, `-l/--lock`, `-x/--exclude`, `-s/--stack N` (QB + N WR/TE same team), `-srb/--stack-rb`, `-te/--max-te`, `-ndo/--no-dst-opp`, `-ms/--max-salary` (Showdown only), `-mns/--min-salary` (both NFL scripts; a salary floor, default 0 = none), `-c/--ceiling` and `-pj/--projceiling` (multi-lineup scripts only; mutually exclusive), `-sf/--small-field` (`NFL-Multi-Opto-v2.0.py` only; picks which ownership projection fills the display-only `Ownership` column), `-ls/--late-swap` (`NFL-Multi-Opto-v2.0.py` only; see below), `-dk/--dk-entries PATH` (both NFL scripts; overrides the entries file, see below).

A single best lineup is `NFL-Multi-Opto-v2.0.py <file> -n 1` — there is no separate single-lineup script.

Showdown specifics: `-l`/`-x` accept an optional slot suffix (`"Drake Maye:CPT"`, `"Sam Darnold:FLEX"`); `-u` counts uniqueness by roster **spot**, so re-using six players with a different Captain is two uniques; `-s`, `-srb`, `-te`, `-ndo` do not apply.

Preserve the argument-reference comment block in each script's docstring — the user relies on it as the CLI cheat sheet:

```python
# Means: python <script> <projections file> -n <number of lineups> -u <min uniques> -e <export to CSV>
```

## Optimization model (the part worth knowing before editing)

All scripts share one pattern: **build the PuLP problem once, then solve it repeatedly, appending a diversity constraint after each solve.** Constraints are never rebuilt per lineup — the same `prob` object accumulates cuts.

- Objective target (both NFL scripts): `OPTIMIZATION_TARGETS` maps a target key to `(label, projection weight, ceiling weight)` and `target_value()` folds those weights into one per-player coefficient, so switching targets changes only the objective — never a constraint. Default is projection-only (`1.0, 0.0`), `--ceiling` is `0.0, 1.0`, `--projceiling` is `0.5, 0.5`. Showdown evaluates the same weights against `CptProjection`/`CptCeiling` for the Captain var and `Projection`/`Ceiling` for the FLEX var. `validate_target_data()` raises when a ceiling-weighted target meets a missing or all-zero `Ceiling` column. The NBA scripts do not carry this — adding it there means porting all four pieces (constants, the two helpers, the argparse group, the objective).

- Classic NFL: one binary var per player. Roster constraints are `QB == 1`, `RB >= 2`, `WR >= 3`, `TE >= 1`, `DST == 1`, `FLEX-eligible == 7`, total `== 9` — the FLEX slot is expressed as that count identity rather than a separate variable. `game_id` is a `frozenset({team, opp})` so both rows of a game map to one id; linking vars enforce "at least two games".
- Showdown: **two** binary vars per player (`cpt_vars[i]`, `flex_vars[i]`) with `cpt + flex <= 1` per player, `sum(cpt) == 1`, `sum(flex) == 5`, and `>= 1` rostered player from each of the two teams. Captain salary/projection/ceiling come from the file's `CPT Salary` / `CPT Proj` / `CPT Ceiling` columns when present, else derived at 1.5x — one precedence for all three, so a source whose Captain values are not exactly 1.5x scales the Captain identically under every optimization target.
- Diversity: `sum(vars of the just-solved lineup) <= ROSTER_SIZE - min_uniques`. Showdown's version is slot-aware — it sums `cpt_vars[captain]` plus the five `flex_vars`, which is why promoting a FLEX to Captain counts as two uniques.
- Display slot assignment is post-hoc, not part of the model: `_assign_flex_positions()` fills RB/WR/TE slots by descending salary and drops the leftover FLEX-eligible player into FLEX. In the classic script, `main()` first maps a `Kickoff` column onto the players by ID from `load_dk_kickoffs()` (the entries file's `Game Info`); when present, the surplus position holds back its latest kickoff for FLEX instead of its cheapest (tie → cheapest). `In Progress` parses to no kickoff and sorts earliest. No file, or no match, reduces to the salary rule; the printed table only shows the `Kickoff (ET)` column when some player matched, and exports drop it via `EXPORT_COLUMNS`.
- Salary floor (`-mns`): 0 adds no constraint at all, so a run without the flag solves exactly the model it always did. Above 0 it is one extra `>= min_salary` constraint next to the cap (Showdown sums the Captain and FLEX salary vars the same way the cap does). Validation is up front: the classic script rejects a floor above `SALARY_CAP`, Showdown rejects one above the clamped `--max-salary`.
- A non-`Optimal` status breaks the loop; lineup #1 failing means the base constraints are infeasible, later failures just mean the pool is exhausted. Both messages name `-mns` when a floor is set, since a tight floor shrinks the pool as readily as it blocks lineup #1.
- Late swap (`-ls`, `run_late_swap()`) is a separate path: `main()` returns after it, never building the model above. It reads the entries file (see below), and a slot is locked when its player's `Game Info` holds no future Eastern kickoff at runtime (or DK tagged `(LOCKED)`; unparseable counts as started). One small problem per entry, **rebuilt each entry**, over the open slots only: the same count identities scoped to the open labels; locked players are constants (salary, games, TE cap, `-u` overlap); stacks/`-ndo` bind new picks only. Diversity is per contest (`Contest ID`), with locked overlap as an unavoidable floor. Fallbacks form a ladder, one rule dropped at a time, `-u` outranking the salary floor: `-u` + floor, `-u` alone, floor alone, neither, then write the entry unchanged. The floor covers the whole entry, so the locked slots' salary counts toward it and only the shortfall is asked of the open slots; a floor the locked players already clear adds no constraint. Projections match the DK pool **by ID**, never name. `_seat_open_slots()` seats post-hoc: pick counts fix which position supplies FLEX, and within it the latest kickoff sits there. `_keep_original_slots()` then leaves each kept player in his original slot and seats only new picks in the vacated ones, unless that puts an earlier kickoff in FLEX — so an unchanged lineup is never reported as a swap. The summary counts swapped (different player set), reseated only, and already optimal separately. Output is `upload-ready-DKEntries[-early|-late]-<timestamp>.csv` in Downloads (the slate tag from `SLATE_FILE_TAGS`, none for Main): every entry, unchanged cells verbatim, new cells the pool's `Name + ID`.

## Input CSV expectations

Classic NFL (`NFL-Multi-Opto-v2.0.py`): headers are resolved through `COLUMN_ALIASES` / `OWNERSHIP_ALIASES` by `resolve_columns()`, not read literally. Comparison is case- and punctuation-insensitive (`_normalize_header` strips to lowercase alphanumerics), and the first alias present wins so a rename can't produce duplicate columns. `ID`←`id`/`DK ID`, `Position`←`DK Pos`/`Pos`, `Salary`←`DK Salary`, `Projection`←`Proj`/`DK Proj`, `Ceiling`←`DK Ceiling`, `Ownership`←`Large Field`/`Own` (under `-sf`: `Small Field` only — the unlabeled legacy headers serve the default request only, so the flag can never return the other field's numbers). Anything still unresolved gets a `difflib` fuzzy pass at `FUZZY_HEADER_CUTOFF` (0.85) that skips claimed headers, every alias of every *other* column (both ownership variants, so `-sf` can't eat `Large Field`), the `UNMATCHABLE_HEADERS` decoys (`DK Value`, `DK Floor`), and alias spellings shorter than `MIN_FUZZY_TARGET_LENGTH` (difflib's `2M/T` makes a fixed cutoff meaningless against a 3-character target: `Down` vs `own` scores 0.857); each fuzzy hit is printed. A `REQUIRED_COLUMNS` entry that stays unresolved raises with the file's actual headers listed. `Ownership` and `Ceiling` are optional and default to 0 with a note, so only the ceiling-weighted targets actually require `Ceiling`. `Opp` may be `"@BUF"` or `"BUF"`. Salary, Projection, Ceiling and Ownership all run the same strip-then-coerce pipeline (`"$6,000 "`, `"11.70%"`, `"1,024.8"`), and `ID` is coerced too, so a malformed value drops the row through `critical_cols` instead of raising at `.astype(int)`. A header the resolver did not choose but that collides with a rename target (a literal `Ownership` column under `-sf`) is dropped with a note, so the rename can never leave two columns sharing one name.

Note that `COLUMN_ALIASES` names two *inverted* shapes: `{internal: (source, ...)}` in the classic multi-lineup script above, `{source: internal}` in the Showdown script below. Same name, opposite direction — read the script's own definition before assuming either.

Showdown: aliases are applied via `COLUMN_ALIASES` (`Pos`, `Proj`, `Total Own`, `Own`, `CPT Own`, `CPT Salary`, `CPT Proj`, `CPT Ceiling`), first alias wins so the rename can't create duplicate columns. `Ceiling`/`Ownership`/`CptOwnership` are optional and default to 0. Ownership is slot-aware: `FlexOwnership = Total Own - CPT Own`, so the printed total is true product ownership. The file must contain exactly two teams or loading raises.

Exports (`-e`) go to `EXPORT_DIR = r"G:\My Drive\Documents\NFL-DFS\csv-exports"` (NBA scripts use their own OneDrive paths), filename timestamped and tagged with the optimization target (`_ceiling` / `_projceiling`) on the NFL multi-lineup scripts. The classic script also tags the slate: `detect_slate()` reads `Main`/`Early`/`Late` from the projections file name (`SLATE_PATTERN`; no match → Main), which labels the lineup headers and late-swap banner, tags the late-swap upload file name, and inserts `SLATE_FILE_TAGS` (`_early`/`_late`, Main empty) after `nfl_classic`. Export columns are unchanged by the target. Each lineup writes one row per roster spot, then a `TOTAL` row, then a DraftKings upload row holding only `Name + ID` values positioned into `EXPORT_COLUMNS[1:]`.

One entries file serves a run: `find_dk_entries_file(args.dk_entries)` returns the `-dk` path (raising if it does not exist — an explicit path is never skipped), else the newest `DKEntries*.csv` in `DOWNLOADS_DIR`, else `None`. In the classic script that one path feeds the upload row, the FLEX kickoffs and late swap (which raises on `None`); Showdown resolves it only under `-e`. `_find_dk_pool()` locates the player pool — the row holding `Name + ID`, anywhere in the file, behind the jagged entry-list columns — and returns a header→column map plus the rows below; every pool reader goes through it. The upload row is built from that file. `load_dk_name_ids()` indexes it three ways — slot+name, name alone (ambiguous names map to `None`), and slot+team for defenses — because Showdown assigns a player **different IDs at CPT and FLEX**. Anything unresolvable (missing file, unmatched player) drops just the upload row; it is never a fatal error. The two NFL multi-lineup scripts carry their own identical copy of these helpers (`_newest_download`, `find_projections_file`, `find_dk_entries_file`, `_find_dk_pool`, `load_dk_name_ids`, and the lookup/upload helpers) — `patch` both or neither.

## PR workflow

`.github/workflows/claude-auto-pr-once.yml` runs a Claude review automatically when a PR opens, and on repo-owner comments containing `@claude`. In remote/cloud sessions, a `@claude` review-request comment on a subscribed PR is a trigger for **that** workflow, not a task for the session — wait for the workflow's review and act on its findings (this rule is injected by `.claude/hooks/pr-review-posture.sh`).

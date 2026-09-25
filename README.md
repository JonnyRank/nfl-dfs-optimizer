# NFL DFS Optimizers

Python optimizers that build DraftKings NFL lineups from a projections CSV. Each optimizer
models the contest as a mixed-integer linear program with [PuLP](https://coin-or.github.io/pulp/)
and maximizes total projected points subject to the salary cap and DraftKings roster rules.

| Script | Format | Roster | Lineups | Solver |
| --- | --- | --- | --- | --- |
| `nfl-classic` | Classic | QB, 2 RB, 3 WR, TE, FLEX, DST (9) | Many | HiGHS |
| `nfl-showdown` | Showdown (Captain Mode) | 1 CPT + 5 FLEX (6) | Many | HiGHS |

The optimizers live in the `nfl_dfs_optimizer` package under `src/`. `uv run nfl-classic` and
`uv run nfl-showdown` run them; the old commands (`uv run nfl-classic`,
`uv run nfl-showdown`) still work and behave identically.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

This creates `.venv` with Python 3.14 and installs `pandas`, `pulp`, `highspy`, and `tzdata`. Both optimizers solve with HiGHS, so `highspy` is required. `tzdata` supplies the Eastern time zone late swap reads kickoffs in (Windows has no built-in zone database).

## Running

Both optimizers find their projections CSV in your Downloads folder on their own, taking the newest
match — re-downloads such as `... (1).csv` included:

* Classic: `DraftKings NFL DFS Projections*.csv` — the Main, Early, and Late slate files all match.
* Showdown: `DK NFL Showdown Projections*.csv`.

To use a different file, pass its full path as the first argument (a file, not a folder). The
file actually used is printed at the start of every run.

**Slates (Classic).** The slate comes from the file name: `Early Slate` and `Late Slate` label
every printed lineup (`--- Optimal NFL Early Slate Lineup #1 ---`) and the late-swap heading, and
add `_early` / `_late` to the export name (`nfl_classic_early_multi_lineups_<timestamp>.csv`).
`Main Slate`, or a file name with no slate in it, runs as Main and keeps the usual export name.
Export contents are the same for every slate.

```bash
# Newest Classic projections in Downloads, 20 lineups
uv run nfl-classic -n 20 -u 2 -e

# Single best Classic lineup
uv run nfl-classic "C:\path\to\projections.csv" -n 1 -e

# 20 Classic lineups, at least 2 players different between any two, QB stacked with a WR/TE,
# no DST opposite one of your own offensive players
uv run nfl-classic "C:\path\to\projections.csv" -n 20 -u 2 -s -ndo -e

# 20 Showdown lineups with a locked Captain and a little salary left on the table
uv run nfl-showdown "C:\path\to\showdown.csv" -n 20 -u 2 -l "Drake Maye:CPT" -ms 49800 -e

# 20 Classic lineups that must spend at least $49,500 of the cap
uv run nfl-classic "C:\path\to\projections.csv" -n 20 -u 2 -mns 49500 -e

# 20 Classic lineups built for upside instead of median points
uv run nfl-classic "C:\path\to\projections.csv" -n 20 -u 2 -c -e

# 20 Showdown lineups on a 50/50 blend of projection and ceiling
uv run nfl-showdown "C:\path\to\showdown.csv" -n 20 -u 2 -pj -e

# Late swap: re-optimize every entry in Downloads\DKEntries.csv around the games already started
uv run nfl-classic "C:\path\to\projections.csv" -ls -u 2

# Use a specific DraftKings entries file instead of the newest one in Downloads
uv run nfl-classic "C:\path\to\projections.csv" -n 20 -e -dk "C:\path\to\DKEntries.csv"
```

Lineups print to the terminal as a formatted table with total projection, ownership, ceiling,
and salary. Nothing is written to disk unless you pass `-e`.

**FLEX seating (Classic).** When there is a DraftKings entries file (see
[The upload row and DKEntries.csv](#the-upload-row-and-dkentriescsv)), the Classic optimizer reads each player's kickoff from its `Game Info` column, matched by
player ID, and seats the player with the latest kickoff in the FLEX — the lineup itself is
unchanged, only who sits where. The position counts still decide which position fills the FLEX
(a third RB means an RB sits there); among that position the latest game takes it. The printed
table gains a `Kickoff (ET)` column (`4:25PM`), and the export and its upload row follow the same
seating. A player whose game already shows `In Progress` has no kickoff: it prints `-` and counts
as the earliest game, so it never takes the FLEX from a player with a kickoff still to come.
Without an entries file the FLEX goes to the cheapest player of that position, as before.

## Options

### `nfl-classic` (Classic, multi-lineup; `legacy/NFL-Multi-Opto-v2.0.py`)

| Flag | Meaning |
| --- | --- |
| `filepath` | Path to the projections CSV (optional; default: newest `DraftKings NFL DFS Projections*.csv` in Downloads) |
| `-n`, `--num-lineups` | Number of lineups to generate (default: 1) |
| `-u`, `--min-uniques` | Minimum players that must differ between any two lineups (default: 1) |
| `-e`, `--export` | Write the lineups to a timestamped CSV |
| `-l`, `--lock` | Player names to force into every lineup, e.g. `-l "Josh Allen" "Puka Nacua"` |
| `-x`, `--exclude` | Player names to keep out of every lineup |
| `-s`, `--stack [N]` | Require the QB to be paired with at least N WR/TE from his own team (N defaults to 1 when the flag is given without a number) |
| `-srb`, `--stack-rb` | Require the QB to be paired with an RB from his own team |
| `-te`, `--max-te` | Cap the number of TEs, e.g. `-te 1` to keep a TE out of the FLEX |
| `-ndo`, `--no-dst-opp` | Never roster a DST alongside a QB/RB/WR/TE from the opposing team |
| `-mns`, `--min-salary` | Require every lineup to spend at least this much salary (default: no floor); applies to `-ls` too |
| `-c`, `--ceiling` | Optimize on ceiling instead of projection |
| `-pj`, `--projceiling` | Optimize on a 50/50 blend of projection and ceiling |
| `-sf`, `--small-field` | Show small-field ownership instead of large-field (display/export only) |
| `-ls`, `--late-swap` | Late-swap your DraftKings entries instead of building new lineups — see [Late swap](#late-swap) |
| `-dk`, `--dk-entries` | DraftKings entries CSV to use instead of the newest `DKEntries*.csv` in Downloads (upload row, FLEX kickoffs, late swap) |

Name matching for `-l` and `-x` is case-insensitive. A name that isn't in the projections file
prints a warning and is skipped rather than failing the run.

### `nfl-showdown` (Showdown, multi-lineup; `legacy/NFL-SD-Multi-Opto-v1.0.py`)

| Flag | Meaning |
| --- | --- |
| `filepath` | Path to the Showdown projections CSV (optional; default: newest `DK NFL Showdown Projections*.csv` in Downloads) |
| `-n`, `--num-lineups` | Number of lineups to generate (default: 1) |
| `-u`, `--min-uniques` | Minimum roster spots that must differ between any two lineups (default: 1) |
| `-e`, `--export` | Write the lineups to a timestamped CSV |
| `-l`, `--lock` | Players to force into every lineup |
| `-x`, `--exclude` | Players to keep out of every lineup |
| `-ms`, `--max-salary` | Cap total lineup salary below $50,000 (values above the cap are clamped) |
| `-mns`, `--min-salary` | Require every lineup to spend at least this much salary (default: no floor); it must not exceed `-ms` |
| `-c`, `--ceiling` | Optimize on ceiling instead of projection |
| `-pj`, `--projceiling` | Optimize on a 50/50 blend of projection and ceiling |
| `-dk`, `--dk-entries` | DraftKings entries CSV for the upload rows, instead of the newest `DKEntries*.csv` in Downloads |

Showdown notes:

* The roster is 1 Captain + 5 FLEX. The Captain scores 1.5x points and costs 1.5x salary, and
  a player can fill the Captain slot or a FLEX slot but never both.
* Every lineup must include at least one player from each of the two teams.
* `-l` and `-x` accept an optional slot suffix: `-l "Drake Maye:CPT"` locks him at Captain,
  `-x "Sam Darnold:CPT"` bans him from Captain but still allows him in the FLEX. Without a
  suffix the lock or exclusion applies to both slots.
* `-u` counts uniqueness by roster spot, so the same six players with a different Captain
  counts as two uniques.
* `-s`, `-srb`, `-te`, and `-ndo` don't apply to Showdown.
* Under any optimization target the Captain contributes its Captain-slot value — the file's
  `CPT Proj` / `CPT Ceiling` when present, otherwise 1.5x the FLEX value.

## Optimization target

Both optimizers maximize **projection** by default. Two mutually exclusive
flags swap in a different scoring target; everything else (salary cap, roster rules, locks,
stacks, diversity) is unchanged.

| Flag | What the solver maximizes |
| --- | --- |
| *(none)* | `Projection` — the median-points lineup, same as always |
| `-c`, `--ceiling` | `Ceiling` — the highest-upside lineup, for GPPs |
| `-pj`, `--projceiling` | `0.5 x Projection + 0.5 x Ceiling` — upside without abandoning floor |

* The run prints `Optimizing on: <target>` after loading, and every lineup still prints its
  projection, ownership, and ceiling totals. A `--projceiling` run also prints its blend score,
  since that number matches neither of the printed totals.
* Ceiling-weighted targets need real ceiling data. If the `Ceiling` column is missing or all
  zeros, the run stops with an explanatory error rather than quietly returning an arbitrary
  salary-feasible lineup. Projection-only runs are unaffected — a missing `Ceiling` column
  just displays as `0.00`.
* A *partly* populated `Ceiling` column still runs, but says so. Blank or unparseable ceilings
  become `0.00`, and the load prints how many players that affected. Under `--ceiling` those
  players can't be rostered unless locked; under `--projceiling` they're scored on projection
  alone. Either way a warning names them, so a quietly shrunken player pool never passes
  unnoticed.
* Exports tag the filename with the target (`..._ceiling_<timestamp>.csv`,
  `..._projceiling_<timestamp>.csv`) so files from different targets don't get mixed up. The
  columns inside the file are unchanged.

## Projections CSV format

### Classic

Required columns: `ID`, `Player`, `Position`, `Team`, `Opp`, `Salary`, `Proj`. `Own` and
`Ceiling` are optional and default to `0.00` (`Ceiling` is required only for `--ceiling` /
`--projceiling`).

The Classic optimizer resolves headers through an alias table instead of taking them
literally, so a projections source that renames its columns loads without hand-editing the CSV:

| Internal column | Accepted headers |
| --- | --- |
| `ID` | `ID`, `id`, `DK ID`, `Player ID` |
| `Player` | `Player`, `Name`, `Player Name` |
| `Position` | `Position`, `DK Pos`, `Pos` |
| `Team` | `Team`, `Tm` |
| `Opp` | `Opp`, `Opponent` |
| `Salary` | `Salary`, `DK Salary` |
| `Projection` | `Projection`, `Proj`, `DK Proj` |
| `Ceiling` | `Ceiling`, `DK Ceiling` |
| `Ownership` | `Large Field`, `Ownership`, `Own` — or, under `-sf`, `Small Field` only |

* Matching ignores case and punctuation, so `id` and `ID` are the same header. The first
  accepted header actually present wins, so two source columns can never collapse onto one
  internal name.
* A header matching nothing in the table falls back to fuzzy matching (85% similarity), and
  every fuzzy resolution is printed. Near-miss decoys (`DK Value`, `DK Floor`) and the
  ownership column you did *not* ask for are excluded from that fallback, so a wrong guess
  can't quietly swap in the wrong numbers.
* The fuzzy pass ignores alias spellings shorter than six characters. `difflib`'s ratio is
  `2M/T` over the combined length, so an 85% cutoff gets weaker the shorter the target: against
  `Own`, any four-letter header containing that run (`Down`, `Town`) scores `0.857` and would
  clear it. Short names are exact-match only, which costs nothing — a header close enough to
  `Tm` or `Opp` to be worth guessing at already hits as an exact alias.
* Only the default (large-field) request accepts the unlabeled legacy `Own` / `Ownership`
  headers, since on the files that carried one it was the only ownership column there was.
  `-sf` is an explicit request for the other measure, so it takes a column that actually says
  `Small Field` or shows `0.00%` with a note — it never falls back to an unlabeled column that
  may hold large-field numbers.
* A required column that stays unresolved raises an error naming it and listing the headers
  the file actually contained — the run never proceeds on a mis-mapped column.
* `Salary` may be formatted (`$6,000`) and ownership may carry a `%` — both are cleaned on load.
* `Opp` may be written `@BUF` or `BUF`; the `@` is stripped when pairing teams into games.
* Rows missing `ID`, `Salary`, `Proj`, or `Position` are dropped before optimizing.
* Every lineup is required to use players from at least two different games.
* A missing `Ceiling` column defaults to zero and displays as `0.00`.

### Showdown

Expected columns: `Player`, `Pos`, `Team`, `Salary`, `Proj`, plus the optional
`Ceiling`, `Total Own`, `CPT Own`, `CPT Salary`, `CPT Proj`, and `CPT Ceiling`.

* The file must contain exactly two teams — filter it down to the single game first, or
  loading fails with an explanatory error.
* `CPT Salary`, `CPT Proj`, and `CPT Ceiling` are used when present; otherwise the standard
  1.5x multiplier is applied to the FLEX values. This matters for `--ceiling`: if your source
  publishes Captain values that aren't exactly 1.5x, supplying `CPT Ceiling` keeps the Captain
  scaled the same way under every target.
* Ownership is slot-aware: the Captain contributes its `CPT Own` and each FLEX contributes
  `Total Own - CPT Own`, so the printed total is true product ownership for the exact lineup.
* Missing `Ceiling` / `Total Own` / `CPT Own` columns default to zero and display as `0.00`.

## Exports

`-e` writes a timestamped file (`nfl_classic_multi_lineups_<timestamp>.csv`,
`nfl_showdown_multi_lineups_<timestamp>.csv`, with `_ceiling` / `_projceiling` inserted before
the timestamp when one of those targets is used, and `_early` / `_late` after `nfl_classic` for
an Early or Late slate) to the directory set by the `EXPORT_DIR`
constant in `src/nfl_dfs_optimizer/common.py` — currently `G:\My Drive\Documents\NFL-DFS\csv-exports`.
Change that constant to export somewhere else.

Each lineup is written as three blocks, all sharing a `Lineup_ID`:

1. **One row per roster spot**, in slot order.
2. **A `TOTAL` row** with the lineup's salary, projection, ownership, and ceiling.
3. **A DraftKings upload row** — the same lineup laid out horizontally, holding only the
   players' `Name + ID` values in slot order, ready to paste into a DraftKings entries file.

```
1,CPT,Drake Maye,QB,NE,15000,28.95,3.63,49.2
1,FLEX,Jaxon Smith-Njigba,WR,SEA,10600,17.9,12.92,30.4
...
1,TOTAL,,,,50000,97.65,78.51,165.9
1,Drake Maye (43782098),Jaxon Smith-Njigba (43782034),...
```

### The upload row and DKEntries.csv

The `Name + ID` values are read from the newest `DKEntries*.csv` in your Downloads folder, so a
browser re-download such as `DKEntries (1).csv` is picked up automatically; `-dk` names a
different file, and a `-dk` path that doesn't exist stops the run. The Classic optimizer uses
the same file for FLEX kickoffs and late swap. That file is jagged: contest entries come first
and the player pool section follows a few rows down with its own header, listing `Position`, `Name + ID`, `Name`,
`ID`, `Roster Position`, `Salary`, `Game Info`, and `TeamAbbrev`.

* Showdown players appear twice — once at `CPT` and once at `FLEX` — with **different IDs**,
  so the Captain row and FLEX rows are looked up by slot and get the correct ID for each.
* Names are matched case-insensitively, ignoring punctuation and generational suffixes, so
  `AJ Brown` in your projections still finds `A.J. Brown` in the DraftKings file. Defenses
  fall back to matching on team abbreviation, since DraftKings names them by nickname.
* **If the file isn't there, the upload row is simply skipped** and the export continues from
  the `TOTAL` row to the next lineup. The same happens for an individual lineup if any of its
  players can't be matched — the optimizer prints which player it couldn't resolve.

## Late swap

`-ls` re-optimizes the lineups you have already entered, mid-slate. Download your entries CSV
from DraftKings' Edit Entries page and run the Classic optimizer with your current projections:

```bash
uv run nfl-classic "C:\path\to\projections.csv" -ls -u 2
```

* **Input** — the newest `DKEntries*.csv` in your Downloads folder, so a browser re-download
  such as `DKEntries (1).csv` is picked up automatically, or the file named with `-dk`.
  Classic entries files only.
* **Who is locked** — a player whose game has started, judged from the file's `Game Info`
  column against the clock when you run late swap (kickoffs are Eastern). `In Progress`
  counts as started, and so does DraftKings' own `(LOCKED)` tag. A locked player stays in his
  slot.
* **Who can swap in** — only players whose games have not started. Unstarted players already
  in a lineup are fair game: the optimizer may keep them, move them to another slot, or
  replace them. Players are matched to your projections by DraftKings ID, so a player with no
  projection can't be swapped in.
* **FLEX** — the new picks' position counts decide which position supplies the FLEX; within
  that position, the player with the latest kickoff sits there, which keeps the most options
  open if you late-swap again later.
* **`-u`** — enforced between entries in the same contest only; entries in different contests
  may end up identical. When locked players alone already make two entries overlap more than
  `-u` allows, the new picks must all differ. If an entry can't meet `-u` at all, it is built
  without it and the optimizer says so.
* **Rules** — `-l`, `-x`, `-s`, `-srb`, `-te`, `-ndo`, `-mns`, `-c`, `-pj` and `-sf` apply to the
  new picks. `-n` and `-e` are ignored. An entry with no swap that fits the cap and your rules is
  written back unchanged, with a note.
* **Salary floor** — `-mns` covers the whole entry, locked players included, so only the salary
  left over is asked of the open slots. An entry that can't reach the floor is swapped without it
  rather than left alone, and the note says so. `-u` outranks the floor: the floor is the first
  rule given up when both can't hold.
* **Output** — `upload-ready-DKEntries-<timestamp>.csv` in Downloads
  (`upload-ready-DKEntries-early-<timestamp>.csv` / `-late-` for those slates), holding every entry
  (changed or not) in DraftKings' upload layout. Each cell is DraftKings' own `Name + ID`
  text, `(LOCKED)` tag included, ready to upload on the Edit Entries page.

Each entry's heading shows Projection, Ceiling and Ownership before -> after the swap, whatever
the target. Each entry prints with a status per slot: `LOCKED` (game started), `KEEP` (same player, same
slot), `MOVE` (already on the lineup, new slot), or `NEW` (swapped in).

Players you keep stay in the slots they already hold. The one exception is FLEX: when the same
players seated differently would put a later kickoff in FLEX, they are reseated, since that keeps
a swap open after the earlier game starts. The summary counts entries as swapped (different
players), reseated only (same players, new slots), or already optimal (your original lineup is
still the best one, so nothing changes).

## When fewer lineups come back than you asked for

Each solved lineup adds a constraint forbidding it from reappearing, so the pool shrinks as
the run goes on. If the first lineup can't be built at all, the constraints are contradictory —
usually conflicting locks, too many exclusions, a `-ms` value that's too low, or a `-mns` floor
that's too high. If the run
stops partway through, the slate has no more lineups that satisfy your `-u` setting or your
`-mns` floor; lower `-u`, lower the floor, or loosen the stacking flags. Both messages name the
floor when one is set.

## Development

```bash
uv run ruff check .   # lint
uv run pytest         # tests
```

CI runs both on every push to main and every pull request.

The parity tests run both optimizers over a matrix of flag combinations and compare each run's
lineup count, per-lineup score and salary, printed output, and export against saved goldens in
`tests/fixtures/golden/`. Their inputs in `tests/fixtures/public/` are obfuscated copies of real
projections (player order preserved, numbers changed). After an intended behavior change,
regenerate the goldens with `uv run python tests/parity_harness.py --regen`.

Code layout: each format has a core module (`src/nfl_dfs_optimizer/classic.py`, `showdown.py`)
that loads projections, validates an options object mirroring the command-line flags, and runs
the optimizer without printing anything; `cli/` holds the command lines that match player names
and print the results; `common.py` holds what both formats share; late swap lives in
`late_swap.py` and is command-line only.

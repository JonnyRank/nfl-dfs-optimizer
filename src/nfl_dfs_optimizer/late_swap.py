"""
Late swap: re-optimizes the entries in a DraftKings entries file mid-slate.

The core, silent like the other cores: it raises for bad input and returns
what it did, and the callers print it (cli/late_swap.py, for -ls) or show it
(the app's Late swap page). See CLAUDE.md ("Late swap") for the model.

A slot is locked when its player's Game Info holds no future Eastern kickoff
at runtime (or DraftKings tagged it LOCKED). One small problem per entry is
rebuilt over the open slots only; diversity is per contest; fallbacks drop one
rule at a time (-u outranks the salary floor). The finished lineups go to the
Downloads folder in DraftKings' own upload layout.

The stages, in the order the CLI prints between them: load_slate() reads the
entries file against the projections, rebuild_entries() re-optimizes the
chosen entries, and write_upload() writes the upload file. run() chains them
for a caller that prints nothing in between.
"""

import csv
import io
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import pulp

from nfl_dfs_optimizer import common
from nfl_dfs_optimizer.classic import (
    ROSTER_SIZE,
    SLATE_FILE_TAGS,
    SLATE_MAIN,
    ClassicOptions,
    _game_info_timezone,
    parse_kickoff,
    target_value,
)
from nfl_dfs_optimizer.common import (
    SALARY_CAP,
    TARGET_PROJECTION,
    _find_dk_pool,
    build_solver,
)

UPLOAD_FILE_PREFIX: str = "upload-ready-DKEntries"
# DraftKings tags a rostered player whose game has started. The clock decides
# on its own, but the tag is honored too: DraftKings rejects any change to a
# player it has already locked, whatever the local clock says.
DK_LOCKED_TAG: str = "(LOCKED)"
DK_ID_PATTERN = re.compile(r"\((\d+)\)")
# Entry ID, Contest Name, Contest ID, and Entry Fee precede the roster slots.
ENTRY_SLOT_START_COL: int = 4
FLEX_SLOT: str = "FLEX"

# EntryOutcome.status values.
SWAPPED = "swapped"  # different players
RESEATED = "reseated"  # same players, new slots
KEPT = "kept"  # the original lineup is still optimal
ALL_LOCKED = "locked"  # every slot locked; nothing to swap
FAILED = "failed"  # no valid swap; entry left unchanged

# Per-slot status in an entry's table.
SLOT_LOCKED, SLOT_KEEP, SLOT_MOVE, SLOT_NEW = "LOCKED", "KEEP", "MOVE", "NEW"


@dataclass(frozen=True)
class LateSwapOptions:
    """
    The Classic CLI flags late swap reads. Locks and excludes are DraftKings
    player IDs. `entry_ids` limits the rebuild to those entries (Entry ID
    text); None rebuilds every entry, as -ls does. Entries left out are not
    rebuilt or written, but still count toward -u within their contest.
    """

    min_uniques: int = 1  # -u, within each contest
    lock_ids: tuple[int, ...] = ()  # -l
    exclude_ids: tuple[int, ...] = ()  # -x
    stack: int = 0  # -s N; 0 = off
    stack_rb: bool = False  # -srb
    max_te: int | None = None  # -te
    no_dst_opp: bool = False  # -ndo
    min_salary: int = 0  # -mns; covers the whole entry, locked players included
    target: str = TARGET_PROJECTION  # -c / -pj
    entry_ids: frozenset[str] | None = None

    def validate(self) -> None:
        """Raises ValueError with the CLI's wording, range checks included."""
        ClassicOptions(
            min_uniques=self.min_uniques,
            stack=self.stack,
            max_te=self.max_te,
            min_salary=self.min_salary,
            target=self.target,
        ).validate(strict=True)
        if self.entry_ids is not None and not self.entry_ids:
            raise ValueError("No entries are selected to late swap.")


@dataclass(frozen=True)
class DkPoolPlayer:
    """One player from the player pool section of a DraftKings entries file."""

    dk_id: int
    name: str
    name_id: str  # DraftKings' "Name + ID" cell, copied verbatim into the upload
    primary_slot: str  # "QB", "RB", "WR", "TE", or "DST"
    flex_eligible: bool
    salary: int
    team: str
    kickoff: datetime | None  # None once Game Info stops carrying a start time
    started: bool


@dataclass(frozen=True)
class DkEntry:
    """One contest entry: its identifying cells and one cell per roster slot."""

    entry_id: str
    contest_name: str
    contest_id: str
    entry_fee: str
    cells: list[str]


def load_dk_entries_file(
    path: str, now: datetime
) -> tuple[list[str], list[str], list[DkEntry], dict[int, DkPoolPlayer]]:
    """
    Reads the contest entries and the player pool from a DraftKings entries file.

    The file is jagged: entries fill the leading columns (Entry ID, Contest
    Name, Contest ID, Entry Fee, then one column per roster slot) and the
    player pool sits to their right under its own "Name + ID" header, a few
    rows down. A pool player counts as started when his Game Info no longer
    shows a future kickoff as of `now`, or when DraftKings has tagged him
    LOCKED; a Game Info the parser cannot read counts as started too, so an
    unreadable time can never let a player be swapped in after his game began.

    Args:
        path: Location of the entries file.
        now: The moment eligibility is judged at (an aware datetime).

    Returns:
        (upload header, slot labels, entries, pool keyed by DraftKings ID).

    Raises:
        ValueError: If the file is not a Classic entries file or has no pool.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise FileNotFoundError(f"Could not read {path}: {exc}") from exc

    filename = os.path.basename(path)
    if not rows or not rows[0] or rows[0][0].strip() != "Entry ID":
        raise ValueError(f"{filename} is not a DraftKings entries file (no 'Entry ID' header).")

    width = ENTRY_SLOT_START_COL + ROSTER_SIZE
    upload_header = rows[0][:width]
    slot_labels: list[str] = []
    for label in rows[0][ENTRY_SLOT_START_COL:]:
        if not label.strip():
            break
        slot_labels.append(label.strip().upper())
    if len(slot_labels) != ROSTER_SIZE:
        raise ValueError(
            f"{filename} has {len(slot_labels)} roster slots "
            f"({', '.join(slot_labels)}); late swap needs a Classic entries file "
            f"with {ROSTER_SIZE}."
        )

    # Entry rows run on past the pool header -- the two blocks share rows --
    # so the whole file is scanned, and a row counts as an entry only when it
    # opens with a numeric Entry ID. A pool row never does, even if a future
    # export shifted the pool block into column 0.
    entries: list[DkEntry] = []
    for cells in rows[1:]:
        if not cells or not cells[0].strip().isdigit():
            continue
        padded = cells[:width] + [""] * (width - len(cells))
        entries.append(
            DkEntry(
                entry_id=padded[0],
                contest_name=padded[1],
                contest_id=padded[2].strip(),
                entry_fee=padded[3],
                cells=padded[ENTRY_SLOT_START_COL:width],
            )
        )
    if not entries:
        raise ValueError(f"{filename} lists no contest entries.")

    found = _find_dk_pool(rows)
    if found is None:
        raise ValueError(f"{filename} has no player pool section ('Name + ID' header).")
    columns, pool_rows = found
    needed = ("Name + ID", "Name", "ID", "Roster Position", "Salary", "Game Info", "TeamAbbrev")
    missing = [name for name in needed if name not in columns]
    if missing:
        raise ValueError(f"{filename}'s player pool is missing column(s): {', '.join(missing)}.")
    col = {name: columns[name] for name in needed}

    # Game Info times are Eastern whatever zone `now` arrives in. A naive
    # `now` passed straight here is taken as Eastern; load_slate() and the app
    # go through eastern_now() first, which reads a naive value as local time.
    zone = _game_info_timezone()
    if now.tzinfo is None:
        now = now.replace(tzinfo=zone)
    pool: dict[int, DkPoolPlayer] = {}
    for cells in pool_rows:
        if len(cells) <= max(col.values()):
            continue
        try:
            dk_id = int(cells[col["ID"]].strip())
            salary = int(cells[col["Salary"]].strip())
        except ValueError:
            continue
        # "RB/FLEX" -> primary slot RB, FLEX-eligible.
        slots = [s.strip().upper() for s in cells[col["Roster Position"]].split("/") if s.strip()]
        if not slots:
            continue
        name_id = cells[col["Name + ID"]]
        kickoff = parse_kickoff(cells[col["Game Info"]], zone)
        pool[dk_id] = DkPoolPlayer(
            dk_id=dk_id,
            name=cells[col["Name"]].strip(),
            name_id=name_id,
            primary_slot=slots[0],
            flex_eligible=FLEX_SLOT in slots[1:],
            salary=salary,
            team=cells[col["TeamAbbrev"]].strip(),
            kickoff=kickoff,
            started=DK_LOCKED_TAG in name_id or kickoff is None or kickoff <= now,
        )
    if not pool:
        raise ValueError(f"{filename}'s player pool has no readable players.")
    return upload_header, slot_labels, entries, pool


def split_entry_slots(
    entry: DkEntry, pool: dict[int, DkPoolPlayer]
) -> tuple[list[int | None], set[int], list[int]]:
    """
    Sorts an entry's slots into locked and open.

    A slot is locked when its player's game has started (or DraftKings tagged
    the cell LOCKED), and also when its ID is missing from the pool -- the
    caller leaves such an entry unchanged, since that player's salary is
    unknown. Blank cells (reservations) and unstarted players are open.

    Returns:
        (DraftKings ID per slot or None, locked slot indices, open slot indices).
    """
    original_ids: list[int | None] = []
    locked: set[int] = set()
    open_slots: list[int] = []
    for idx, cell in enumerate(entry.cells):
        match = DK_ID_PATTERN.search(cell)
        dk_id = int(match.group(1)) if match else None
        original_ids.append(dk_id)
        if dk_id is None:
            open_slots.append(idx)
            continue
        player = pool.get(dk_id)
        if player is None or player.started or DK_LOCKED_TAG in cell:
            locked.add(idx)
        else:
            open_slots.append(idx)
    return original_ids, locked, open_slots


def optimize_late_swap_entry(
    open_labels: list[str],
    locked_ids: list[int],
    pool: dict[int, DkPoolPlayer],
    candidates: dict[int, DkPoolPlayer],
    projections: dict[int, dict[str, Any]],
    target: str,
    options: LateSwapOptions,
    lock_ids: set[int],
    previous: list[frozenset[int]],
    min_salary: int,
    solver: pulp.LpSolver,
) -> list[int] | None:
    """
    Picks the best players for one entry's open slots.

    Same count-identity model as the full optimizer, sized to the open slots:
    each open positional slot needs a player of that position, and the open
    FLEX (if any) takes exactly one surplus RB/WR/TE. Locked players are fixed
    constants -- their salary comes off the cap, they count toward the games
    and TE limits, and they count toward shared players under -u -- but the
    stacking and DST rules bind new picks only: a locked player's teammates
    and opponents have started too, so no new pick could change those rules'
    outcome for him.

    Args:
        open_labels: Roster slot label of each open slot ("RB", "FLEX", ...).
        locked_ids: DraftKings IDs of the entry's locked players.
        pool: Every player in the entries file, keyed by DraftKings ID.
        candidates: Players eligible to be swapped in (unstarted, projected,
            not excluded).
        projections: Projection rows keyed by DraftKings ID.
        target: Optimization target key.
        options: The rules (stack, stack_rb, max_te, no_dst_opp,
            min_uniques).
        lock_ids: Locked player IDs; forced in wherever they fit an open slot.
        previous: Lineups already built for this contest, for -u.
        min_salary: Salary floor for the whole entry, 0 for none. The locked
            players' salary counts toward it, so only the shortfall is asked
            of the open slots.
        solver: The HiGHS solver from build_solver(), shared by every entry.

    Returns:
        The chosen DraftKings IDs (one per open slot, unordered), or None when
        no valid set exists.
    """
    open_counts = Counter(label for label in open_labels if label != FLEX_SLOT)
    open_flex = len(open_labels) - sum(open_counts.values())
    flex_positions = {p.primary_slot for p in pool.values() if p.flex_eligible}
    locked_set = set(locked_ids)

    # Only players who fit one of the open slots get a variable. A player
    # already fixed in a locked slot never does -- DraftKings' LOCKED tag can
    # lock a player whose kickoff the clock still calls future, and picking
    # him again would roster him twice.
    fits = {
        dk_id: player
        for dk_id, player in candidates.items()
        if dk_id not in locked_set
        and (player.primary_slot in open_counts or (open_flex and player.flex_eligible))
    }

    def members(predicate) -> list[int]:
        return [dk_id for dk_id, player in fits.items() if predicate(player)]

    prob = pulp.LpProblem("DraftKings_NFL_Late_Swap", pulp.LpMaximize)
    pick = pulp.LpVariable.dicts("Pick", list(fits), cat="Binary")
    prob += pulp.lpSum(target_value(projections[i], target) * pick[i] for i in fits)

    locked_players = [pool[i] for i in locked_ids]
    locked_salary = sum(p.salary for p in locked_players)
    open_salary = pulp.lpSum(fits[i].salary * pick[i] for i in fits)
    prob += (open_salary <= SALARY_CAP - locked_salary, "Salary_Cap")
    # The floor covers the whole entry, so the locked slots' salary counts
    # toward it; a floor they already clear constrains nothing.
    if min_salary - locked_salary > 0:
        prob += (open_salary >= min_salary - locked_salary, "Min_Salary")
    prob += pulp.lpSum(pick[i] for i in fits) == len(open_labels), "Open_Slots"

    # Every open positional slot needs its own position; the FLEX surplus must
    # be FLEX-eligible. With the slot total above, these two force exactly one
    # surplus RB/WR/TE per open FLEX and no surplus anywhere else.
    for slot, count in open_counts.items():
        eligible = members(lambda p, slot=slot: p.primary_slot == slot)
        if len(eligible) < count:
            return None
        prob += pulp.lpSum(pick[i] for i in eligible) >= count, f"Min_{slot}"
    flex_needed = open_flex + sum(c for s, c in open_counts.items() if s in flex_positions)
    flex_members = members(lambda p: p.flex_eligible)
    if len(flex_members) < flex_needed:
        return None
    if flex_members:
        prob += pulp.lpSum(pick[i] for i in flex_members) == flex_needed, "FLEX_Logic"

    # At least two games across the whole lineup. Started and unstarted games
    # never overlap, so locked games simply add to the count.
    def game_key(dk_id: int) -> Any:
        if dk_id in projections:
            return projections[dk_id]["game_id"]
        return frozenset([pool[dk_id].team])

    locked_games = {game_key(i) for i in locked_ids}
    if len(locked_games) < 2:
        games: dict[Any, list[int]] = defaultdict(list)
        for i in fits:
            games[game_key(i)].append(i)
        game_vars = pulp.LpVariable.dicts("Game", range(len(games)), cat="Binary")
        for g_idx, game_members in enumerate(games.values()):
            # A game counts only when someone from it is actually picked.
            prob += game_vars[g_idx] <= pulp.lpSum(pick[i] for i in game_members)
        prob += (
            pulp.lpSum(game_vars.values()) >= 2 - len(locked_games),
            "At_Least_Two_Games",
        )

    if options.max_te is not None:
        locked_te = sum(1 for p in locked_players if p.primary_slot == "TE")
        tes = members(lambda p: p.primary_slot == "TE")
        if tes:
            # An open TE slot must be filled whatever the cap says -- a locked
            # FLEX TE under -te 1 would otherwise make the entry infeasible.
            te_allowance = max(options.max_te - locked_te, open_counts.get("TE", 0))
            prob += (
                pulp.lpSum(pick[i] for i in tes) <= te_allowance,
                "Max_TE_Constraint",
            )

    def team(dk_id: int) -> str:
        return str(projections[dk_id]["Team"])

    for qb in members(lambda p: p.primary_slot == "QB"):
        if options.stack > 0:
            mates = [
                i for i in fits
                if team(i) == team(qb) and fits[i].primary_slot in ("WR", "TE")
            ]
            prob += pulp.lpSum(pick[i] for i in mates) >= options.stack * pick[qb]
        if options.stack_rb:
            mates = [i for i in fits if team(i) == team(qb) and fits[i].primary_slot == "RB"]
            prob += pulp.lpSum(pick[i] for i in mates) >= pick[qb]

    if options.no_dst_opp:
        for dst in members(lambda p: p.primary_slot == "DST"):
            opp = str(projections[dst]["Opp"]).replace("@", "")
            for off in members(lambda p: p.primary_slot != "DST"):
                if team(off) == opp:
                    prob += pick[dst] + pick[off] <= 1

    for dk_id in lock_ids & fits.keys():
        prob += pick[dk_id] == 1

    # -u against the lineups already built for this contest. Locked overlap is
    # fixed, so where it alone exceeds the allowance the best achievable is no
    # further overlap among the new picks.
    for prev in previous:
        shared = [i for i in fits if i in prev]
        if shared:
            allowance = ROSTER_SIZE - options.min_uniques - len(locked_set & prev)
            prob += pulp.lpSum(pick[i] for i in shared) <= max(allowance, 0)

    prob.solve(solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return [i for i in fits if pick[i].varValue > 0.5]


def _seat_open_slots(
    open_slots: list[int], slot_labels: list[str], picks: list[DkPoolPlayer]
) -> dict[int, DkPoolPlayer] | None:
    """
    Seats the solver's picks in an entry's open slots.

    The model chooses players by position count, so seating is post-hoc, as
    in the full optimizer. The picks' position counts fix which position
    supplies the FLEX; within that position, earlier kickoffs take the
    positional slots and the latest kickoff goes to FLEX. A FLEX player can
    be swapped for any RB/WR/TE on a later late-swap run, so this keeps the
    most options open that the picks allow.

    Returns:
        Slot index -> player, or None if the picks cannot fill the slots.
    """
    positional: dict[str, list[int]] = defaultdict(list)
    flex_slots: list[int] = []
    for idx in open_slots:
        if slot_labels[idx] == FLEX_SLOT:
            flex_slots.append(idx)
        else:
            positional[slot_labels[idx]].append(idx)

    by_position: dict[str, list[DkPoolPlayer]] = defaultdict(list)
    for player in picks:
        by_position[player.primary_slot].append(player)

    seated: dict[int, DkPoolPlayer] = {}
    surplus: list[DkPoolPlayer] = []
    for position, players in by_position.items():
        # Earliest kickoff first; within one kickoff, highest salary first
        # (the full optimizer's order), so the cheapest late player is surplus.
        players.sort(key=lambda p: (p.kickoff.timestamp() if p.kickoff else 0.0, -p.salary))
        slots = positional.get(position, [])
        seated.update(zip(slots, players))
        surplus.extend(players[len(slots) :])
    if len(surplus) != len(flex_slots) or any(not p.flex_eligible for p in surplus):
        return None
    seated.update(zip(flex_slots, surplus))
    return seated if len(seated) == len(open_slots) else None


def _keep_original_slots(
    open_slots: list[int],
    slot_labels: list[str],
    original_ids: list[int | None],
    picks: list[DkPoolPlayer],
    seated: dict[int, DkPoolPlayer],
) -> dict[int, DkPoolPlayer]:
    """
    Leaves kept players in the slots they already hold where that costs nothing.

    _seat_open_slots() orders players by kickoff and salary, so on its own it
    can shuffle two kept WRs between slots and report an unchanged lineup as a
    swap. This seats every kept player in his original slot and only the new
    picks in the slots that were vacated. That seating wins unless it puts an
    earlier kickoff in FLEX than `seated` does -- a later FLEX is worth a real
    reorder, since it keeps a swap open once the earlier game has started.

    Returns:
        The stay-put seating when it is valid and gives up no FLEX kickoff,
        else `seated` unchanged.
    """
    pick_ids = {p.dk_id for p in picks}
    stay_put = {
        idx: next(p for p in picks if p.dk_id == original_ids[idx])
        for idx in open_slots
        if original_ids[idx] in pick_ids
    }
    kept_ids = {p.dk_id for p in stay_put.values()}
    vacated = [idx for idx in open_slots if idx not in stay_put]
    fill = _seat_open_slots(
        vacated, slot_labels, [p for p in picks if p.dk_id not in kept_ids]
    )
    if fill is None:
        return seated
    stay_put.update(fill)

    def flex_kickoffs(seating: dict[int, DkPoolPlayer]) -> list[float]:
        return sorted(
            p.kickoff.timestamp() if p.kickoff else 0.0
            for idx, p in seating.items()
            if slot_labels[idx] == FLEX_SLOT
        )

    return stay_put if flex_kickoffs(stay_put) >= flex_kickoffs(seated) else seated


@dataclass
class LateSwapSlate:
    """An entries file read for late swap, with the projections matched to it by ID."""

    entries_path: str
    now: datetime  # the Eastern moment "started" was judged at
    upload_header: list[str]
    slot_labels: list[str]
    entries: list[DkEntry]
    pool: dict[int, DkPoolPlayer]
    projections: dict[int, dict[str, Any]]  # projection rows keyed by DraftKings ID

    @property
    def contests(self) -> set[str]:
        return {entry.contest_id for entry in self.entries}

    def check_projections(self) -> None:
        """Raises ValueError when no projection matches a pool player by ID."""
        if not self.pool.keys() & self.projections.keys():
            raise ValueError(
                f"No projections ID matches a DraftKings ID in "
                f"{os.path.basename(self.entries_path)}. Late swap matches players by "
                f"DraftKings ID, so the projections file must carry DraftKings player "
                f"IDs in its ID column."
            )

    def started_lock_names(self, lock_ids: set[int]) -> list[str]:
        """Locked players whose games have started, by name."""
        return sorted(
            self.pool[i].name for i in lock_ids if i in self.pool and self.pool[i].started
        )

    def candidates(self, exclude_ids: set[int]) -> dict[int, DkPoolPlayer]:
        """Players eligible to be swapped in: unstarted, projected, not excluded."""
        return {
            dk_id: player
            for dk_id, player in self.pool.items()
            if not player.started and dk_id in self.projections and dk_id not in exclude_ids
        }

    def started_count(self) -> int:
        return sum(1 for p in self.pool.values() if p.started)

    def unprojected_names(self) -> list[str]:
        """Unstarted pool players with no projection, who cannot be swapped in."""
        return sorted(
            p.name for p in self.pool.values() if not p.started and p.dk_id not in self.projections
        )


def eastern_now(now: datetime | None = None) -> datetime:
    """`now` in Eastern, the zone Game Info is written in; default: the current time."""
    zone = _game_info_timezone()
    return now.astimezone(zone) if now else datetime.now(zone)


def load_slate(
    players_df: pd.DataFrame, entries_path: str, now: datetime | None = None
) -> LateSwapSlate:
    """
    Reads an entries file for late swap as of `now` (default: the current
    time), matching `players_df` (from load_player_data(), ownership field
    applied) to its pool by DraftKings ID. Does not check that anything
    matched; see LateSwapSlate.check_projections().

    Raises:
        FileNotFoundError / ValueError: From load_dk_entries_file().
    """
    now = eastern_now(now)
    upload_header, slot_labels, entries, pool = load_dk_entries_file(entries_path, now)
    projections = (
        players_df.drop_duplicates("ID").set_index("ID", drop=False).to_dict("index")
    )
    return LateSwapSlate(entries_path, now, upload_header, slot_labels, entries, pool, projections)


def unprojected_note(names: list[str]) -> str:
    """The note naming unstarted players that have no projection."""
    shown = ", ".join(names[:5])
    if len(names) > 5:
        shown += f", +{len(names) - 5} more"
    return (
        f"{len(names)} unstarted player(s) have no projection "
        f"and cannot be swapped in: {shown}"
    )


def started_locks_note(names: list[str]) -> str:
    return (
        f"Games have started for {', '.join(names)}; they "
        f"stay only in the entries that already roster them."
    )


def pool_summary(slate: LateSwapSlate, candidates: dict[int, DkPoolPlayer]) -> str:
    return (
        f"{slate.started_count()} of {len(slate.pool)} pool players' games have started; "
        f"{len(candidates)} players are eligible to swap in."
    )


@dataclass
class EntryOutcome:
    """One entry's late swap: what it held, what it holds now, and why."""

    # 1-based among the entries rebuilt, not the entry's row in the file:
    # with entries skipped, "Entry 3/10" is the third one rebuilt.
    number: int
    entry: DkEntry
    original_ids: list[int | None]
    final_ids: list[int | None]
    locked: set[int]  # locked slot indices
    status: str  # SWAPPED, RESEATED, KEPT, ALL_LOCKED or FAILED
    note: str = ""


def rebuild_entries(
    slate: LateSwapSlate,
    options: LateSwapOptions,
    candidates: dict[int, DkPoolPlayer],
    solver: pulp.LpSolver,
) -> list[EntryOutcome]:
    """
    Re-optimizes the chosen entries (options.entry_ids, else all), in file
    order. Players whose games have started stay in their slots; every other
    slot is refilled from `candidates`, and each entry must differ by -u
    players from the entries already settled in its contest -- every entry
    left out of the rebuild, and the rebuilt ones before it. An entry with no
    valid swap is kept unchanged.
    """
    lock_ids = set(options.lock_ids)
    chosen = [
        entry
        for entry in slate.entries
        if options.entry_ids is None or entry.entry_id in options.entry_ids
    ]
    history: dict[str, list[frozenset[int]]] = defaultdict(list)
    # Entries not being rebuilt stay as they are, so every rebuilt entry in
    # the same contest must differ from them wherever they sit in the file.
    for entry in slate.entries:
        if options.entry_ids is not None and entry.entry_id not in options.entry_ids:
            original_ids = split_entry_slots(entry, slate.pool)[0]
            history[entry.contest_id].append(frozenset(i for i in original_ids if i is not None))

    outcomes: list[EntryOutcome] = []
    for number, entry in enumerate(chosen, start=1):
        original_ids, locked, open_slots = split_entry_slots(entry, slate.pool)
        locked_ids = [original_ids[idx] for idx in sorted(locked)]
        final_ids = list(original_ids)
        note = ""
        unknown = [
            entry.cells[idx] for idx in sorted(locked) if original_ids[idx] not in slate.pool
        ]

        if unknown:
            status = FAILED
            note = (
                f"{', '.join(unknown)} not in the player pool, so the salary "
                f"left under the cap is unknown; entry left unchanged."
            )
        elif not open_slots:
            status = ALL_LOCKED
            note = "every slot is locked; nothing to swap."
        else:
            open_labels = [slate.slot_labels[idx] for idx in open_slots]
            previous = history[entry.contest_id]
            seated: dict[int, DkPoolPlayer] | None = None

            # The full rules first, then one rule at a time dropped: -u
            # outranks the salary floor, so the floor is given up first, and
            # the entry is only left unchanged when nothing solves at all.
            attempts = [
                (prev, floor)
                for prev in ([previous, []] if previous else [[]])
                for floor in ([options.min_salary, 0] if options.min_salary else [0])
            ]
            for prev, floor in attempts:
                picks = optimize_late_swap_entry(
                    open_labels, locked_ids, slate.pool, candidates, slate.projections,
                    options.target, options, lock_ids, prev, floor, solver,
                )
                if picks is None:
                    continue
                pick_players = [slate.pool[i] for i in picks]
                seated = _seat_open_slots(open_slots, slate.slot_labels, pick_players)
                if seated is not None:
                    seated = _keep_original_slots(
                        open_slots, slate.slot_labels, original_ids, pick_players, seated
                    )
                if seated is None:
                    # The count identities guarantee a seating, so this means
                    # the model and the seater disagree -- not infeasibility.
                    note = (
                        "the optimizer found a lineup the seating step could not "
                        "place (a bug); entry left unchanged."
                    )
                else:
                    dropped = []
                    if previous and not prev:
                        dropped.append(
                            f"could not differ by {options.min_uniques} from every "
                            f"earlier entry in this contest"
                        )
                    if options.min_salary and not floor:
                        dropped.append(
                            f"could not reach the ${options.min_salary:,} salary floor"
                        )
                    if dropped:
                        rules = "those rules" if len(dropped) > 1 else "that rule"
                        note = f"{'; '.join(dropped)}; built without {rules}."
                break

            if seated is None:
                status = FAILED
                note = note or (
                    "no swap fits the salary cap and your rules "
                    "(-l/-x/-s/-srb/-te/-ndo); entry left unchanged."
                )
            else:
                for idx, player in seated.items():
                    final_ids[idx] = player.dk_id
                # A swap means different players. The same players in new
                # slots is only a reseat, and an unchanged entry was kept.
                if set(final_ids) != set(original_ids):
                    status = SWAPPED
                else:
                    # Appended, not a fallback: a dropped-rule note must not
                    # hide that the entry came back unchanged.
                    if final_ids != original_ids:
                        status = RESEATED
                        outcome = "same players; the later kickoff moved into FLEX."
                    else:
                        status = KEPT
                        outcome = "the original lineup is still optimal; no change."
                    note = f"{note} {outcome}" if note else outcome

        history[entry.contest_id].append(frozenset(i for i in final_ids if i is not None))
        outcomes.append(EntryOutcome(number, entry, original_ids, final_ids, locked, status, note))
    return outcomes


def summarize(outcomes: list[EntryOutcome]) -> str:
    """The one-line count of what the rebuild did."""
    counts = Counter(outcome.status for outcome in outcomes)
    return (
        f"Late swap complete: {counts[SWAPPED]} of {len(outcomes)} entries swapped, "
        f"{counts[RESEATED]} reseated only, {counts[KEPT]} already optimal, "
        f"{counts[ALL_LOCKED]} fully locked, {counts[FAILED]} with no valid swap."
    )


# --- Showing an entry ---


def _stat(slate: LateSwapSlate, dk_id: int | None, column: str) -> float:
    row = slate.projections.get(dk_id) if dk_id is not None else None
    return float(row[column]) if row else 0.0


def entry_totals(slate: LateSwapSlate, outcome: EntryOutcome) -> dict[str, tuple[float, float]]:
    """Projection, Ceiling and Ownership as (before, after), whatever the target."""
    return {
        column: (
            sum(_stat(slate, i, column) for i in outcome.original_ids),
            sum(_stat(slate, i, column) for i in outcome.final_ids),
        )
        for column in ("Projection", "Ceiling", "Ownership")
    }


def entry_salary(slate: LateSwapSlate, outcome: EntryOutcome) -> int:
    """The entry's salary after the swap."""
    return sum(slate.pool[i].salary for i in outcome.final_ids if i is not None and i in slate.pool)


def entry_rows(slate: LateSwapSlate, outcome: EntryOutcome) -> list[dict[str, Any] | None]:
    """
    One dict per roster slot, in the file's slot order (None for an empty
    slot): Slot, Player, Pos, Team, Salary, Proj, Own%, Ceiling, and Status
    (LOCKED, KEEP in its slot, MOVE to another slot, or NEW).
    """
    original_set = {i for i in outcome.original_ids if i is not None}
    rows: list[dict[str, Any] | None] = []
    for idx, dk_id in enumerate(outcome.final_ids):
        if dk_id is None:
            rows.append(None)
            continue
        if idx in outcome.locked:
            status = SLOT_LOCKED
        elif dk_id == outcome.original_ids[idx]:
            status = SLOT_KEEP
        elif dk_id in original_set:
            status = SLOT_MOVE
        else:
            status = SLOT_NEW
        player = slate.pool.get(dk_id)
        rows.append(
            {
                "Slot": slate.slot_labels[idx],
                "Player": player.name if player else outcome.entry.cells[idx],
                "Pos": player.primary_slot if player else "",
                "Team": player.team if player else "",
                "Salary": player.salary if player else 0,
                "Proj": _stat(slate, dk_id, "Projection"),
                "Own%": _stat(slate, dk_id, "Ownership"),
                "Ceiling": _stat(slate, dk_id, "Ceiling"),
                "Status": status,
            }
        )
    return rows


# --- The upload file ---


def upload_row(slate: LateSwapSlate, outcome: EntryOutcome) -> list[str]:
    """
    An entry's upload row. A cell whose player did not change is copied
    verbatim; a new one gets the pool's "Name + ID" text, exactly as
    DraftKings wrote it.
    """
    entry = outcome.entry
    cells = [
        entry.cells[idx] if dk_id == outcome.original_ids[idx] else slate.pool[dk_id].name_id
        for idx, dk_id in enumerate(outcome.final_ids)
    ]
    return [entry.entry_id, entry.contest_name, entry.contest_id, entry.entry_fee, *cells]


def upload_csv(slate: LateSwapSlate, outcomes: list[EntryOutcome]) -> str:
    """The upload file's text: DraftKings' header, then one row per rebuilt entry."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(slate.upload_header)
    writer.writerows(upload_row(slate, outcome) for outcome in outcomes)
    return buffer.getvalue()


def upload_file_name(slate_name: str = SLATE_MAIN) -> str:
    """
    upload-ready-DKEntries-<timestamp>.csv. Early/Late slates are named in the
    file ("upload-ready-DKEntries-early-<timestamp>.csv") so two slates'
    uploads are told apart; Main is unchanged.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005 -- local time, like the exports
    name_parts = [UPLOAD_FILE_PREFIX, SLATE_FILE_TAGS[slate_name].lstrip("_"), timestamp]
    return "-".join(part for part in name_parts if part) + ".csv"


def write_upload(
    slate: LateSwapSlate,
    outcomes: list[EntryOutcome],
    slate_name: str = SLATE_MAIN,
    directory: str | None = None,
) -> str:
    """Writes the upload file to `directory` (default: Downloads) and returns its path."""
    folder = directory if directory is not None else common.DOWNLOADS_DIR
    output_path = os.path.join(folder, upload_file_name(slate_name))
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        handle.write(upload_csv(slate, outcomes))
    return output_path


# --- All at once ---


@dataclass
class LateSwapResult:
    """A whole late swap: the slate read, each rebuilt entry, and the notes."""

    slate: LateSwapSlate
    outcomes: list[EntryOutcome]
    messages: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return summarize(self.outcomes)


def run(
    players_df: pd.DataFrame,
    entries_path: str,
    options: LateSwapOptions,
    now: datetime | None = None,
) -> LateSwapResult:
    """
    Late-swaps the chosen entries of `entries_path` as of `now` (default: the
    current time). Writes nothing; see write_upload().

    Raises:
        ValueError / FileNotFoundError: Invalid options, an unreadable
            entries file, projections that match no pool player by ID, or no
            HiGHS solver.
    """
    options.validate()
    # Checked before any file is read, so a missing highspy stops the run
    # rather than surfacing once entries are already being rebuilt.
    solver = build_solver()
    slate = load_slate(players_df, entries_path, now)
    slate.check_projections()
    if options.entry_ids is not None:
        unknown = options.entry_ids - {entry.entry_id for entry in slate.entries}
        if unknown:
            raise ValueError(
                f"{os.path.basename(entries_path)} has no entry {', '.join(sorted(unknown))}."
            )
    messages: list[str] = []
    started = slate.started_lock_names(set(options.lock_ids))
    if started:
        messages.append(started_locks_note(started))
    candidates = slate.candidates(set(options.exclude_ids))
    messages.append(pool_summary(slate, candidates))
    unprojected = slate.unprojected_names()
    if unprojected:
        messages.append(unprojected_note(unprojected))
    outcomes = rebuild_entries(slate, options, candidates, solver)
    return LateSwapResult(slate, outcomes, messages)

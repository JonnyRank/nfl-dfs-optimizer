"""
Late swap (-ls): re-optimizes the entries in a DraftKings entries file mid-slate.

CLI-only, and moved here verbatim from the pre-package Classic script: it still
takes the parsed argparse Namespace and prints as it goes. See CLAUDE.md
("Late swap") for the model.

A slot is locked when its player's Game Info holds no future Eastern kickoff
at runtime (or DraftKings tagged it LOCKED). One small problem per entry is
rebuilt over the open slots only; diversity is per contest; fallbacks drop one
rule at a time (-u outranks the salary floor). The finished lineups go to the
Downloads folder in DraftKings' own upload layout.
"""

# Moved verbatim from the legacy script, so its typing style is left as it was.
# ruff: noqa: UP006, UP035, UP045, DTZ005

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd
import pulp

from nfl_dfs_optimizer import common
from nfl_dfs_optimizer.classic import (
    ROSTER_SIZE,
    SLATE_FILE_TAGS,
    SLATE_MAIN,
    _game_info_timezone,
    parse_kickoff,
    target_value,
)
from nfl_dfs_optimizer.common import (
    DK_ENTRIES_GLOB,
    SALARY_CAP,
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


def find_dk_entries_file(override: Optional[str] = None) -> Optional[str]:
    """common.find_dk_entries_file(), printing its note as the script always did."""
    notes: List[str] = []
    path = common.find_dk_entries_file(override, notes=notes)
    for note in notes:
        print(note)
    return path


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
    kickoff: Optional[datetime]  # None once Game Info stops carrying a start time
    started: bool


@dataclass(frozen=True)
class DkEntry:
    """One contest entry: its identifying cells and one cell per roster slot."""

    entry_id: str
    contest_name: str
    contest_id: str
    entry_fee: str
    cells: List[str]



def load_dk_entries_file(
    path: str, now: datetime
) -> Tuple[List[str], List[str], List[DkEntry], Dict[int, DkPoolPlayer]]:
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
    slot_labels: List[str] = []
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
    entries: List[DkEntry] = []
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

    # Game Info times are Eastern whatever zone `now` arrives in; a naive
    # `now` is taken as Eastern too.
    zone = _game_info_timezone()
    if now.tzinfo is None:
        now = now.replace(tzinfo=zone)
    pool: Dict[int, DkPoolPlayer] = {}
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


def _split_entry_slots(
    entry: DkEntry, pool: Dict[int, DkPoolPlayer]
) -> Tuple[List[Optional[int]], Set[int], List[int]]:
    """
    Sorts an entry's slots into locked and open.

    A slot is locked when its player's game has started (or DraftKings tagged
    the cell LOCKED), and also when its ID is missing from the pool -- the
    caller leaves such an entry unchanged, since that player's salary is
    unknown. Blank cells (reservations) and unstarted players are open.

    Returns:
        (DraftKings ID per slot or None, locked slot indices, open slot indices).
    """
    original_ids: List[Optional[int]] = []
    locked: Set[int] = set()
    open_slots: List[int] = []
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


def _ids_for_names(players_df: pd.DataFrame, names: Optional[List[str]], action: str) -> Set[int]:
    """Resolves -l / -x player names (case-insensitive) to DraftKings IDs."""
    ids: Set[int] = set()
    for name in names or []:
        matches = players_df[players_df["Player"].str.lower() == name.lower()]
        if matches.empty:
            print(f"  WARNING: Player '{name}' not found in projections. Skipping {action}.")
            continue
        ids.update(int(dk_id) for dk_id in matches["ID"])
    return ids


def optimize_late_swap_entry(
    open_labels: List[str],
    locked_ids: List[int],
    pool: Dict[int, DkPoolPlayer],
    candidates: Dict[int, DkPoolPlayer],
    projections: Dict[int, Dict[str, Any]],
    target: str,
    args: argparse.Namespace,
    lock_ids: Set[int],
    previous: List[FrozenSet[int]],
    min_salary: int,
    solver: pulp.LpSolver,
) -> Optional[List[int]]:
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
        args: Parsed CLI options (stack, stack_rb, max_te, no_dst_opp,
            min_uniques).
        lock_ids: IDs from -l; forced in wherever they fit an open slot.
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

    def members(predicate) -> List[int]:
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
        games: Dict[Any, List[int]] = defaultdict(list)
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

    if args.max_te is not None:
        locked_te = sum(1 for p in locked_players if p.primary_slot == "TE")
        tes = members(lambda p: p.primary_slot == "TE")
        if tes:
            # An open TE slot must be filled whatever the cap says -- a locked
            # FLEX TE under -te 1 would otherwise make the entry infeasible.
            te_allowance = max(args.max_te - locked_te, open_counts.get("TE", 0))
            prob += (
                pulp.lpSum(pick[i] for i in tes) <= te_allowance,
                "Max_TE_Constraint",
            )

    def team(dk_id: int) -> str:
        return str(projections[dk_id]["Team"])

    for qb in members(lambda p: p.primary_slot == "QB"):
        if args.stack > 0:
            mates = [
                i for i in fits
                if team(i) == team(qb) and fits[i].primary_slot in ("WR", "TE")
            ]
            prob += pulp.lpSum(pick[i] for i in mates) >= args.stack * pick[qb]
        if args.stack_rb:
            mates = [i for i in fits if team(i) == team(qb) and fits[i].primary_slot == "RB"]
            prob += pulp.lpSum(pick[i] for i in mates) >= pick[qb]

    if args.no_dst_opp:
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
            allowance = ROSTER_SIZE - args.min_uniques - len(locked_set & prev)
            prob += pulp.lpSum(pick[i] for i in shared) <= max(allowance, 0)

    prob.solve(solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return [i for i in fits if pick[i].varValue > 0.5]


def _seat_open_slots(
    open_slots: List[int], slot_labels: List[str], picks: List[DkPoolPlayer]
) -> Optional[Dict[int, DkPoolPlayer]]:
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
    positional: Dict[str, List[int]] = defaultdict(list)
    flex_slots: List[int] = []
    for idx in open_slots:
        if slot_labels[idx] == FLEX_SLOT:
            flex_slots.append(idx)
        else:
            positional[slot_labels[idx]].append(idx)

    by_position: Dict[str, List[DkPoolPlayer]] = defaultdict(list)
    for player in picks:
        by_position[player.primary_slot].append(player)

    seated: Dict[int, DkPoolPlayer] = {}
    surplus: List[DkPoolPlayer] = []
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
    open_slots: List[int],
    slot_labels: List[str],
    original_ids: List[Optional[int]],
    picks: List[DkPoolPlayer],
    seated: Dict[int, DkPoolPlayer],
) -> Dict[int, DkPoolPlayer]:
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

    def flex_kickoffs(seating: Dict[int, DkPoolPlayer]) -> List[float]:
        return sorted(
            p.kickoff.timestamp() if p.kickoff else 0.0
            for idx, p in seating.items()
            if slot_labels[idx] == FLEX_SLOT
        )

    return stay_put if flex_kickoffs(stay_put) >= flex_kickoffs(seated) else seated


def _print_late_swap_entry(
    number: int,
    total: int,
    entry: DkEntry,
    slot_labels: List[str],
    final_ids: List[Optional[int]],
    original_ids: List[Optional[int]],
    locked: Set[int],
    pool: Dict[int, DkPoolPlayer],
    projections: Dict[int, Dict[str, Any]],
    note: str,
) -> None:
    """Prints one late-swapped entry with a per-slot LOCKED/KEEP/MOVE/NEW status."""

    def stat(dk_id: Optional[int], column: str) -> float:
        row = projections.get(dk_id) if dk_id is not None else None
        return float(row[column]) if row else 0.0

    def change(label: str, before: float, after: float, unit: str = "") -> str:
        return f"{label}: {before:.2f}{unit} -> {after:.2f}{unit} ({after - before:+.2f}{unit})"

    def column_total(ids: List[Optional[int]], column: str) -> float:
        return sum(stat(i, column) for i in ids)

    present = [i for i in final_ids if i is not None]
    salary = sum(pool[i].salary for i in present if i in pool)
    # Projection, Ceiling and Ownership always show before -> after, whatever
    # the target: points on one line, Ownership and Salary on the next.
    points = [
        change(column, column_total(original_ids, column), column_total(final_ids, column))
        for column in ("Projection", "Ceiling")
    ]
    ownership = change(
        "Ownership",
        column_total(original_ids, "Ownership"),
        column_total(final_ids, "Ownership"),
        "%",
    )
    print(f"\n--- Entry {number}/{total}: {entry.entry_id} | {entry.contest_name} ---")
    print(" | ".join(points))
    print(f"{ownership} | Salary: ${salary:,}")
    if note:
        print(f"NOTE: {note}")
    print("-" * 97)
    print(
        f"{'Slot':<5} {'Player':<25} {'Pos':<5} {'Team':<5} "
        f"{'Salary':>8} {'Proj':>8} {'Own%':>8} {'Ceiling':>8}  Status"
    )
    print("-" * 97)
    original_set = {i for i in original_ids if i is not None}
    for idx, dk_id in enumerate(final_ids):
        if dk_id is None:
            print(f"{slot_labels[idx]:<5} - EMPTY -")
            continue
        if idx in locked:
            status = "LOCKED"
        elif dk_id == original_ids[idx]:
            status = "KEEP"
        elif dk_id in original_set:
            status = "MOVE"
        else:
            status = "NEW"
        player = pool.get(dk_id)
        name = player.name if player else entry.cells[idx]
        position = player.primary_slot if player else ""
        team = player.team if player else ""
        player_salary = player.salary if player else 0
        print(
            f"{slot_labels[idx]:<5} {name[:25]:<25} {position:<5} {team:<5} "
            f"${player_salary:>7,} {stat(dk_id, 'Projection'):>8.2f} "
            f"{stat(dk_id, 'Ownership'):>7.2f}% {stat(dk_id, 'Ceiling'):>8.2f}  {status}"
        )
    print("-" * 97)


def run_late_swap(
    args: argparse.Namespace,
    players_df: pd.DataFrame,
    target: str,
    now: Optional[datetime] = None,
    slate: str = SLATE_MAIN,
) -> Optional[str]:
    """
    Late-swaps every entry in the DraftKings entries file (-dk, else the
    newest one in Downloads).

    Players whose games have started stay in their slots. Every other slot is
    re-optimized from the players whose games have not started, and each
    entry must differ from the entries already built for the same contest by
    -u players. An entry with no valid swap is written back unchanged. The
    result is an upload-ready-DKEntries-<timestamp>.csv in Downloads (with
    "-early" / "-late" before the timestamp for those slates) holding every
    entry, each cell DraftKings' own "Name + ID" text.

    Args:
        args: Parsed CLI options.
        players_df: Projections from load_player_data(); matched by DK ID.
        target: Optimization target key.
        now: Moment eligibility is judged at; defaults to the current time.
        slate: The projections file's slate, shown in the heading.

    Returns:
        Path of the upload file written, or None if nothing was written.
    """
    zone = _game_info_timezone()
    now = now.astimezone(zone) if now else datetime.now(zone)

    # Checked before any file is read, so a missing highspy stops the run
    # rather than surfacing once entries are already being rebuilt.
    solver = build_solver()

    entries_path = find_dk_entries_file(args.dk_entries)
    if entries_path is None:
        raise FileNotFoundError(
            f"No {DK_ENTRIES_GLOB} file found in {common.DOWNLOADS_DIR}. Download your "
            f"entries CSV from DraftKings' Edit Entries page first, or name one "
            f"with -dk."
        )
    upload_header, slot_labels, entries, pool = load_dk_entries_file(entries_path, now)
    contests = {entry.contest_id for entry in entries}
    print(f"\n--- Late Swap ({slate} Slate) ---")
    print(f"Entries file: {entries_path}")
    print(
        f"Clock: {now:%m/%d/%Y %I:%M%p} ET. {len(entries)} entries across "
        f"{len(contests)} contest(s)."
    )
    if args.min_salary:
        print(
            f"Salary floor: ${args.min_salary:,} per entry, locked players "
            f"included; dropped for an entry that cannot reach it."
        )
    if args.num_lineups != 1 or args.export:
        print(
            "NOTE: -n and -e do not apply to --late-swap; every entry is "
            "re-optimized and written to the upload file."
        )

    projections = (
        players_df.drop_duplicates("ID").set_index("ID", drop=False).to_dict("index")
    )
    if not pool.keys() & projections.keys():
        raise ValueError(
            f"No projections ID matches a DraftKings ID in {os.path.basename(entries_path)}. "
            f"Late swap matches players by DraftKings ID, so the projections file "
            f"must carry DraftKings player IDs in its ID column."
        )

    if args.lock:
        print(f"\nLocking players: {args.lock}")
    lock_ids = _ids_for_names(players_df, args.lock, "lock")
    started_locks = sorted(pool[i].name for i in lock_ids if i in pool and pool[i].started)
    if started_locks:
        print(
            f"NOTE: Games have started for {', '.join(started_locks)}; they "
            f"stay only in the entries that already roster them."
        )
    if args.exclude:
        print(f"\nExcluding players: {args.exclude}")
    exclude_ids = _ids_for_names(players_df, args.exclude, "exclusion")

    candidates = {
        dk_id: player
        for dk_id, player in pool.items()
        if not player.started and dk_id in projections and dk_id not in exclude_ids
    }
    unprojected = sorted(
        p.name for p in pool.values() if not p.started and p.dk_id not in projections
    )
    started_count = sum(1 for p in pool.values() if p.started)
    print(
        f"{started_count} of {len(pool)} pool players' games have started; "
        f"{len(candidates)} players are eligible to swap in."
    )
    if unprojected:
        names = ", ".join(unprojected[:5])
        if len(unprojected) > 5:
            names += f", +{len(unprojected) - 5} more"
        print(
            f"NOTE: {len(unprojected)} unstarted player(s) have no projection "
            f"and cannot be swapped in: {names}"
        )

    history: Dict[str, List[FrozenSet[int]]] = defaultdict(list)
    upload_rows: List[List[str]] = []
    changed = reseated = kept = unchanged_locked = failed = 0

    for number, entry in enumerate(entries, start=1):
        original_ids, locked, open_slots = _split_entry_slots(entry, pool)
        locked_ids = [original_ids[idx] for idx in sorted(locked)]
        final_ids = list(original_ids)
        note = ""
        unknown = [entry.cells[idx] for idx in sorted(locked) if original_ids[idx] not in pool]

        if unknown:
            failed += 1
            note = (
                f"{', '.join(unknown)} not in the player pool, so the salary "
                f"left under the cap is unknown; entry left unchanged."
            )
        elif not open_slots:
            unchanged_locked += 1
            note = "every slot is locked; nothing to swap."
        else:
            open_labels = [slot_labels[idx] for idx in open_slots]
            previous = history[entry.contest_id]
            seated: Optional[Dict[int, DkPoolPlayer]] = None

            # The full rules first, then one rule at a time dropped: -u
            # outranks the salary floor, so the floor is given up first, and
            # the entry is only left unchanged when nothing solves at all.
            attempts = [
                (prev, floor)
                for prev in ([previous, []] if previous else [[]])
                for floor in ([args.min_salary, 0] if args.min_salary else [0])
            ]
            for prev, floor in attempts:
                picks = optimize_late_swap_entry(
                    open_labels, locked_ids, pool, candidates, projections,
                    target, args, lock_ids, prev, floor, solver,
                )
                if picks is None:
                    continue
                pick_players = [pool[i] for i in picks]
                seated = _seat_open_slots(open_slots, slot_labels, pick_players)
                if seated is not None:
                    seated = _keep_original_slots(
                        open_slots, slot_labels, original_ids, pick_players, seated
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
                            f"could not differ by {args.min_uniques} from every "
                            f"earlier entry in this contest"
                        )
                    if args.min_salary and not floor:
                        dropped.append(
                            f"could not reach the ${args.min_salary:,} salary floor"
                        )
                    if dropped:
                        rules = "those rules" if len(dropped) > 1 else "that rule"
                        note = f"{'; '.join(dropped)}; built without {rules}."
                break

            if seated is None:
                failed += 1
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
                    changed += 1
                else:
                    # Appended, not a fallback: a dropped-rule note must not
                    # hide that the entry came back unchanged.
                    if final_ids != original_ids:
                        reseated += 1
                        outcome = "same players; the later kickoff moved into FLEX."
                    else:
                        kept += 1
                        outcome = "the original lineup is still optimal; no change."
                    note = f"{note} {outcome}" if note else outcome

        history[entry.contest_id].append(
            frozenset(i for i in final_ids if i is not None)
        )
        # A cell whose player did not change is copied verbatim; a new one gets
        # the pool's "Name + ID" text, exactly as DraftKings wrote it.
        cells = [
            entry.cells[idx] if dk_id == original_ids[idx] else pool[dk_id].name_id
            for idx, dk_id in enumerate(final_ids)
        ]
        upload_rows.append(
            [entry.entry_id, entry.contest_name, entry.contest_id, entry.entry_fee, *cells]
        )
        _print_late_swap_entry(
            number, len(entries), entry, slot_labels, final_ids, original_ids,
            locked, pool, projections, note,
        )

    print(
        f"\nLate swap complete: {changed} of {len(entries)} entries swapped, "
        f"{reseated} reseated only, {kept} already optimal, "
        f"{unchanged_locked} fully locked, {failed} with no valid swap."
    )

    # Early/Late slates are named in the file ("upload-ready-DKEntries-early-
    # <timestamp>.csv") so two slates' uploads are told apart; Main is unchanged.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name_parts = [UPLOAD_FILE_PREFIX, SLATE_FILE_TAGS[slate].lstrip("_"), timestamp]
    output_path = os.path.join(
        common.DOWNLOADS_DIR, "-".join(part for part in name_parts if part) + ".csv"
    )
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(upload_header)
        writer.writerows(upload_rows)
    print(f"Upload file written to: {output_path}")
    return output_path

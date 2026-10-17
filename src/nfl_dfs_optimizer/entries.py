"""
Contest entries: reads the entry list of a DraftKings entries file
(DKEntries*.csv) of either format, and writes the upload file that fills
those entries with lineups.

The core, silent like the others: it raises ValueError/FileNotFoundError for
bad input. Late swap reads its entries through parse_entries(); the app's
Classic and Showdown pages fill entries with generated lineups through
read_entries() and upload_csv().
"""

import csv
import io
import os
import re
from dataclasses import dataclass

# Entry ID, Contest Name, Contest ID, and Entry Fee precede the roster slots.
ENTRY_SLOT_START_COL: int = 4
DK_ID_PATTERN = re.compile(r"\((\d+)\)")
# "Josh Allen (44220217) (LOCKED)" -> "Josh Allen".
_CELL_SUFFIX_PATTERN = re.compile(r"(\s*\((\d+|LOCKED)\))+\s*$")


@dataclass(frozen=True)
class DkEntry:
    """One contest entry: its identifying cells and one cell per roster slot."""

    entry_id: str
    contest_name: str
    contest_id: str
    entry_fee: str
    cells: list[str]


@dataclass(frozen=True)
class EntriesFile:
    """An entries file's entry list."""

    header: list[str]  # Entry ID through the last roster slot: the upload file's header
    slot_labels: list[str]  # "QB", "RB", ... or "CPT", "FLEX", ...
    entries: list[DkEntry]


def read_rows(path: str) -> list[list[str]]:
    """Every row of a CSV file, jagged as written."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return list(csv.reader(handle))
    except OSError as exc:
        raise FileNotFoundError(f"Could not read {path}: {exc}") from exc


def parse_entries(rows: list[list[str]], filename: str) -> EntriesFile:
    """
    The entry list of an entries file's rows. The roster slots are the header
    cells after Entry Fee up to the first blank one. The entries may be empty;
    the caller decides whether that is an error.

    Raises:
        ValueError: If the first row is not DraftKings' "Entry ID" header.
    """
    if not rows or not rows[0] or rows[0][0].strip() != "Entry ID":
        raise ValueError(f"{filename} is not a DraftKings entries file (no 'Entry ID' header).")

    slot_labels: list[str] = []
    for label in rows[0][ENTRY_SLOT_START_COL:]:
        if not label.strip():
            break
        slot_labels.append(label.strip().upper())
    width = ENTRY_SLOT_START_COL + len(slot_labels)

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
    return EntriesFile(rows[0][:width], slot_labels, entries)


def read_entries(path: str, roster_size: int | None = None, format_name: str = "") -> EntriesFile:
    """
    Reads an entries file's entry list.

    Args:
        path: Location of the entries file.
        roster_size: When given, the number of roster slots the file must have.
        format_name: The format named when the slot count is wrong ("Classic").

    Raises:
        FileNotFoundError: The file can't be read.
        ValueError: It is not an entries file, has the wrong number of roster
            slots, or lists no entries.
    """
    filename = os.path.basename(path)
    found = parse_entries(read_rows(path), filename)
    if roster_size is not None and len(found.slot_labels) != roster_size:
        raise ValueError(
            f"{filename} has {len(found.slot_labels)} roster slots "
            f"({', '.join(found.slot_labels)}); {format_name or 'this format'} lineups need "
            f"an entries file with {roster_size}."
        )
    if not found.entries:
        raise ValueError(f"{filename} lists no contest entries.")
    return found


def cell_name(cell: str) -> str:
    """A roster cell's player name, without its ID or LOCKED tag ("" for a blank cell)."""
    return _CELL_SUFFIX_PATTERN.sub("", cell).strip()


def upload_csv(found: EntriesFile, values: dict[str, list[str]]) -> str:
    """
    The upload file's text: DraftKings' header, then one row per entry in
    `values` (Entry ID -> one "Name + ID" per roster slot), in file order. An
    entry's identifying cells are copied verbatim.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(found.header)
    for entry in found.entries:
        if entry.entry_id in values:
            writer.writerow(
                [
                    entry.entry_id,
                    entry.contest_name,
                    entry.contest_id,
                    entry.entry_fee,
                    *values[entry.entry_id],
                ]
            )
    return buffer.getvalue()

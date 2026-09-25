"""Helpers both CLIs use."""

import os


def reject_stray_projections_path(filepath: str | None, *name_lists: list[str] | None) -> None:
    """
    Stops a run whose projections path was typed after -l / -x.

    Those flags take one or more values, so argparse files a path written
    after them under the flag and leaves `filepath` empty. Left alone, the run
    would skip the "player" and quietly optimize on the newest download.

    Raises:
        ValueError: If `filepath` is empty and a lock/exclude entry names a
            CSV or an existing file.
    """
    if filepath:
        return
    for names in name_lists:
        for name in names or []:
            if name.lower().endswith(".csv") or os.path.isfile(name):
                raise ValueError(
                    f"'{name}' was read as a player name for -l/-x, but it looks "
                    f"like a projections file. Put the projections path first, "
                    f"before any flags."
                )


def print_notes(notes: list[str]) -> None:
    for note in notes:
        print(note)

"""
Builds the committable parity fixtures in tests/fixtures/public/ from the real
inputs in tests/fixtures/inputs/ (gitignored: the projections are a paid
subscription's data).

Every projected number -- projection, ceiling, floor, value, both ownership
fields, Showdown's Captain projection and ownership -- is replaced by a
rank-preserving perturbation: each value is scaled by random per-player
noise, and the resulting values are then handed back out in the
original column's order. Sorting any column gives the same player order as the
source (ties are broken at random), but no value matches it. Exact zeros stay
zero, so a DST's 0.0% ownership is still 0.0%.

Names, IDs, positions, teams and salaries are public DraftKings data and are
kept. The DKEntries files carry nothing proprietary and are copied verbatim.

Run from the repo root after replacing the real inputs:

    uv run python tests/fixtures/make_public_fixtures.py

The output is committed and is the input the parity goldens were built from,
so regenerating it means regenerating the goldens too.
"""

import csv
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs")
PUBLIC = os.path.join(HERE, "public")
SEED = 20260924

CLASSIC = "DraftKings NFL DFS Projections -- Main Slate.csv"
SHOWDOWN = "DK NFL Showdown Projections.csv"
ENTRIES = ("DKEntriesClassic.csv", "DKEntriesShowdown.csv")


def _parse(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[\$,%\s]", "", regex=True), errors="coerce"
    )


def _perturb(values: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Rank-preserving noise: new values, same order, distinct to two decimals."""
    out = values.copy().astype(float)
    mask = values.notna() & (values != 0)
    if not mask.any():
        return out
    original = values[mask].to_numpy(dtype=float)
    # Multiplicative noise never moves a value across zero, so the sorted
    # noisy values hold exactly as many negatives as the source column.
    noisy = original * np.exp(rng.normal(0, 0.08, len(original)))
    # Hand the noisy values back out in the original order; random tie-breaks.
    order = np.lexsort((rng.random(len(original)), original))
    new = np.empty_like(original)
    new[order] = np.round(np.sort(noisy), 2)
    # Rounding can collapse neighbors; nudge them apart so no tie survives.
    for rank in range(1, len(order)):
        prev, cur = order[rank - 1], order[rank]
        if new[cur] <= new[prev]:
            new[cur] = round(new[prev] + 0.01, 2)
    if (np.sign(new) != np.sign(original)).any():
        raise ValueError("Perturbation flipped a sign; change SEED or the noise.")
    out[mask] = new
    return out


def _fmt(values: pd.Series, suffix: str = "") -> pd.Series:
    return values.map(lambda v: "" if pd.isna(v) else f"{v:.2f}{suffix}")


def _write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(
        os.path.join(PUBLIC, name), index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig"
    )


def build_classic(rng: np.random.Generator) -> None:
    df = pd.read_csv(os.path.join(INPUTS, CLASSIC), dtype=str, encoding="utf-8-sig")
    for col in ("DK Proj", "DK Value", "DK Floor", "DK Ceiling"):
        df[col] = _fmt(_perturb(_parse(df[col]), rng))
    for col in ("Small Field", "Large Field"):
        df[col] = _fmt(_perturb(_parse(df[col]), rng), "%")
    _write(df, CLASSIC)


def build_showdown(rng: np.random.Generator) -> None:
    df = pd.read_csv(os.path.join(INPUTS, SHOWDOWN), dtype=str, encoding="utf-8-sig")
    # Each column is perturbed on its own, CPT Proj and CPT Own included, so
    # every column keeps its own order and the Captain values stop being an
    # exact multiple of the FLEX ones -- the file-supplied Captain path stays
    # distinguishable from the 1.5x fallback.
    for col in ("Proj", "Ceiling", "CPT Proj"):
        df[col] = _fmt(_perturb(_parse(df[col]), rng))
    for col in ("Total Own", "CPT Own"):
        df[col] = _fmt(_perturb(_parse(df[col]), rng), "%")
    _write(df, SHOWDOWN)


def main() -> None:
    os.makedirs(PUBLIC, exist_ok=True)
    rng = np.random.default_rng(SEED)
    build_classic(rng)
    build_showdown(rng)
    for name in ENTRIES:
        shutil.copyfile(os.path.join(INPUTS, name), os.path.join(PUBLIC, name))
    print(f"Wrote public fixtures to {PUBLIC}")


if __name__ == "__main__":
    main()

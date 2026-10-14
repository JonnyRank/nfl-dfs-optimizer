# Performance baseline

Measured with `scripts/bench_perf.py` (Item 1 of the performance push), before any
solver changes: PuLP builds a fresh HiGHS model on every solve.

    uv run python scripts/bench_perf.py             # newest real files in Downloads
    uv run python scripts/bench_perf.py --fixtures  # committed fixtures

Columns: **build** = model construction once per run; per solve, **rebuild** = PuLP
creating a new `highspy.Highs` and re-adding every column and row plus reading the
solution back, **solve** = `Highs.run()` itself, **other** = the rest of the loop
(lineup extraction, FLEX seating, the PuLP diversity cut).

## Baseline (2026-10-14, real Week 5 files)

```
Python 3.14.2 on Windows-11-10.0.26200-SP0, 20 logical CPUs; pulp 3.3.2, highspy 1.15.1, pandas 3.0.6
classic: DraftKings NFL DFS Projections -- Main Slate-Week-5.csv
showdown: DK NFL Showdown Projections.csv
entries: DKEntries-Main-Swap.csv
```

Classic pool 301 players, Showdown pool 29 players.

| Format | n | u | built | total s | build ms | rebuild ms/it | solve ms/it | other ms/it | rebuild % | solve % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classic | 20 | 1 | 20 | 5.15 | 10 | 14.8 | 234.1 | 8.1 | 6% | 91% |
| Classic | 20 | 2 | 20 | 7.44 | 10 | 15.0 | 348.7 | 7.8 | 4% | 94% |
| Classic | 20 | 3 | 20 | 7.66 | 10 | 14.9 | 359.8 | 8.0 | 4% | 94% |
| Classic | 150 | 1 | 150 | 142.19 | 10 | 15.6 | 924.6 | 7.7 | 2% | 98% |
| Classic | 150 | 2 | 150 | 184.42 | 10 | 15.5 | 1206.3 | 7.7 | 1% | 98% |
| Classic | 150 | 3 | 150 | 266.42 | 11 | 15.3 | 1753.2 | 7.5 | 1% | 99% |
| Classic | 500 | 1 | 500 | 1120.02 | 10 | 18.9 | 2212.6 | 8.5 | 1% | 99% |
| Classic | 500 | 2 | 500 | 1309.90 | 10 | 16.3 | 2596.1 | 7.4 | 1% | 99% |
| Classic | 500 | 3 | 500 | 2322.10 | 10 | 16.2 | 4620.5 | 7.4 | 0% | 99% |
| Showdown | 20 | 1 | 20 | 6.51 | 4 | 2.7 | 321.2 | 1.4 | 1% | 99% |
| Showdown | 20 | 2 | 20 | 6.82 | 4 | 2.7 | 336.7 | 1.4 | 1% | 99% |
| Showdown | 20 | 3 | 20 | 5.84 | 4 | 2.7 | 287.5 | 1.4 | 1% | 99% |
| Showdown | 150 | 1 | 150 | 117.37 | 4 | 3.1 | 777.9 | 1.5 | 0% | 99% |
| Showdown | 150 | 2 | 150 | 125.73 | 4 | 3.1 | 833.6 | 1.5 | 0% | 99% |
| Showdown | 150 | 3 | 150 | 84.71 | 4 | 3.1 | 560.1 | 1.4 | 1% | 99% |
| Showdown | 500 | 1 | 500 | 533.05 | 3 | 4.2 | 1060.4 | 1.4 | 0% | 99% |
| Showdown | 500 | 2 | 500 | 541.73 | 4 | 4.2 | 1077.8 | 1.4 | 0% | 99% |
| Showdown | 500 | 3 | 500 | 529.70 | 4 | 4.2 | 1053.7 | 1.5 | 0% | 99% |

Late swap, -u 2, clock 10/11/2026 03:55PM ET
| Entries file | entries | contests | total s | ms/entry | solves | rebuild s | solve s | other s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DKEntries-Main-Swap.csv | 10 | 3 | 0.15 | 15 | 10 | 0.02 | 0.11 | 0.03 |
| synthetic 20x4 | 20 | 4 | 0.28 | 14 | 20 | 0.04 | 0.21 | 0.03 |
| synthetic 60x6 | 60 | 6 | 1.06 | 18 | 60 | 0.13 | 0.88 | 0.05 |
| synthetic 150x10 | 150 | 10 | 10.77 | 72 | 150 | 0.32 | 10.34 | 0.11 |

## Reading

- HiGHS solve time is 91-99% of every multi-lineup run; PuLP's per-solve rebuild is
  15 ms (Classic) / 3-4 ms (Showdown) and roughly flat as cuts accumulate. Replacing
  it with a persistent model caps out near a 1-6% saving.
- Solve time per lineup grows with the cut count (Classic -u 3: 360 ms at n = 20,
  4.6 s average at n = 500), so long runs are dominated by the MIP search itself.
- Late swap at real sizes (10 entries, 3 contests) takes 0.15 s in total.
- Late swap time per entry is flat from 10 to 60 entries (14-18 ms), then reaches 72 ms
  at 150 entries over 10 contests, almost all of it HiGHS: each entry carries a -u cut
  for every earlier entry in its contest. The synthetic files cycle the same few lineups,
  so near-duplicates make those cuts bite harder than a real file's would.

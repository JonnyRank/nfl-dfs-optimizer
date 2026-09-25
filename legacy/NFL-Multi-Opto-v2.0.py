"""
DraftKings NFL Multi-Lineup Optimizer (Classic).

The optimizer lives in the nfl_dfs_optimizer package (src/); this script is a
shim so existing commands keep working. Full documentation and the argument
parser: src/nfl_dfs_optimizer/cli/classic.py.

Input Arguments:
    python NFL-Multi-Opto-v2.0.py ["path"] -n -u -e -te -x -l -s -srb -ndo -mns -c -pj -sf -ls -dk
    python <script> <proj file (optional)> <# of lineups> <min uniques> <max TE> <exclude> <export to CSV> <lock players> <stack QB with WR/TE> <stack QB with RB> <no DST vs Opp> <minimum salary> <optimize on ceiling> <optimize on 50/50 proj+ceiling> <use small-field ownership> <late swap DKEntries.csv> <DKEntries file path>
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -e -l "Josh Allen" -s -ndo
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -c
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -pj
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -mns 49500
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -ls -u 2 -mns 49500
    python NFL-Multi-Opto-v2.0.py "C:\\path\\to\\projections.csv" -n 5 -e -dk "C:\\path\\to\\DKEntries.csv"
    python NFL-Multi-Opto-v2.0.py -n 5 -u 2 -e
"""

from nfl_dfs_optimizer.cli.classic import main

if __name__ == "__main__":
    main()

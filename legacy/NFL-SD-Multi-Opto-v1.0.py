"""
DraftKings NFL Showdown (Captain Mode) Multi-Lineup Optimizer.

The optimizer lives in the nfl_dfs_optimizer package (src/); this script is a
shim so existing commands keep working. Full documentation and the argument
parser: src/nfl_dfs_optimizer/cli/showdown.py.

Input Arguments:
    python NFL-SD-Multi-Opto-v1.0.py ["path"] -n -u -e -l -x -ms -mns -c -pj -dk
    python <script> <proj file (optional)> <# of lineups> <min uniques> <export to CSV> <lock players> <exclude players> <max salary> <minimum salary> <optimize on ceiling> <optimize on 50/50 proj+ceiling> <DKEntries file path>
    # Means: python <script> <projections file> -n <number of lineups> -u <min uniques> -e <export to CSV>
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -e -l "Drake Maye:CPT" -ms 49800
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -ms 49800 -mns 49000
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -c
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -u 2 -pj
    python NFL-SD-Multi-Opto-v1.0.py "C:\\path\\to\\projections.csv" -n 5 -e -dk "C:\\path\\to\\DKEntries.csv"
    python NFL-SD-Multi-Opto-v1.0.py -n 5 -u 2 -e
"""

from nfl_dfs_optimizer.cli.showdown import main

if __name__ == "__main__":
    main()

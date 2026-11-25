# Replication code: exact wild cluster bootstrap for Table 5

Files
- `sequential_contest_bids.csv`: analysis file, 3,160 rows (158 participants × 20 rounds, six sessions).
- `build_dataset.py`: builds the analysis file from the raw oTree exports. Input paths are set inside the script; edit them to point at the raw session folders.
- `table5_bootstrap.py`: exact (all 6^6 = 46,656 patterns) wild cluster bootstrap p-values for every Table 5 coefficient that carries a significance level. Writes `table5_bootstrap_p_values.txt`.

Run
    python3 table5_bootstrap.py sequential_contest_bids.csv

Requirements: Python 3 with numpy, pandas and scipy. Run time is about 40 minutes on two cores.

Method
- Participant random-effects GLS (Swamy–Arora), with session-clustered standard errors and the small-sample factor G/(G−1)·(N−1)/(N−K).
- Restricted wild cluster bootstrap: for each test, the null model drops the tested regressor; bootstrap outcomes are the restricted fitted values plus Webb six-point weights (±√1.5, ±1, ±√0.5) times the restricted residuals, one weight per session.
- p-value = share of the 46,656 weight patterns whose |t| is at least the observed |t|.

Reparameterization (Table 5, columns 4–5)
- Baseline winner effect = Winner × (1 − COST − PRODUCTIVE); winner–loser gap within COST = Winner × COST; within PRODUCTIVE = Winner × PRODUCTIVE. These are the "Winner Effect" rows.
- Treatment effect for winners = COST + COST × Winner, tested through the CostW and ProdW regressors.

Not included
- The Table 1 earnings rows, the Table 4 randomization tests, and the appendix tables. Those are produced by the other scripts in the replication folder.
- R: the script is in Python. The R version has not been run against these files.

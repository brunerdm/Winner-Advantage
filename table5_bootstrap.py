"""
Exact wild cluster bootstrap p-values for Table 5 (Behavior in Repeated Competition with Winner Advantages).

Model: participant random-effects GLS (Swamy-Arora), standard errors clustered on the six sessions.
Inference: restricted wild cluster bootstrap, Webb six-point weights, sessions as clusters,
all 6^6 = 46,656 weight patterns (exact, no simulation). Each test is run on the restricted
model that drops the tested regressor.

Usage:
    python3 table5_bootstrap.py sequential_contest_bids.csv
Output:
    table5_bootstrap_p_values.txt (one line per test; compare with Table 5)
Run time: about 40 minutes on two cores.
"""
import sys, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats

PATH = sys.argv[1] if len(sys.argv) > 1 else "sequential_contest_bids.csv"
OUT = "table5_bootstrap_p_values.txt"

d = pd.read_csv(PATH)
d["UID"] = d["UID"].astype(int)
d["Total"] = d.Bid1 + d.Bid2
d["Winner"] = d.Winner1
d["WinProb1"] = np.where(d.Bid1 + d.opponent_bid1 > 0, d.Bid1 / (d.Bid1 + d.opponent_bid1), 0.0)
d["CxW"] = d.Cost * d.Winner1                       # COST x Winner (original parameterization)
d["PxW"] = d.Productive * d.Winner1
d["Wbase"] = d.Winner1 * (1 - d.Cost - d.Productive)  # reparameterization: baseline winner effect
d["WcostE"] = d.Winner1 * d.Cost                    # winner-loser gap within COST = Winner + Cost x Winner
d["WprodE"] = d.Winner1 * d.Productive
d["CostL"] = d.Cost * (1 - d.Winner1)               # losers' COST effect
d["CostW"] = d.Cost * d.Winner1                     # winners' COST effect (COST + COST x Winner)
d["ProdL"] = d.Productive * (1 - d.Winner1)
d["ProdW"] = d.Productive * d.Winner1

d = d.sort_values(["UID", "Round"]).reset_index(drop=True)
idx = pd.factorize(d.UID)[0]; N = idx.max() + 1; NT = len(d)
cnt = np.bincount(idx); Tbar = NT / N
cl = pd.factorize(d.session)[0]; G = cl.max() + 1
W6 = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
COMBOS = np.array(list(itertools.product(range(6), repeat=G)))   # 46,656 x 6

def design(cols):
    return np.column_stack([np.ones(NT)] + [d[c].to_numpy(float) for c in cols])

def re_fit(yv, X):
    """Swamy-Arora random-effects GLS with session-clustered standard errors (small-sample factor G/(G-1)*(N-1)/(N-K))."""
    K = X.shape[1] - 1
    Xm = np.column_stack([np.bincount(idx, weights=X[:, j]) / cnt for j in range(X.shape[1])])
    ym = np.bincount(idx, weights=yv) / cnt
    Xw = (X - Xm[idx])[:, 1:]; yw = yv - ym[idx]
    bw = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    s2e = np.sum((yw - Xw @ bw) ** 2) / (NT - N - K)
    bb = np.linalg.lstsq(Xm, ym, rcond=None)[0]
    s2b = np.sum((ym - Xm @ bb) ** 2) / (N - K - 1)
    s2u = max(0.0, s2b - s2e / Tbar)
    theta = 1 - np.sqrt(s2e / (s2e + Tbar * s2u))
    Xs = X - theta * Xm[idx]; ys = yv - theta * ym[idx]
    XtXi = np.linalg.inv(Xs.T @ Xs)
    b = XtXi @ Xs.T @ ys
    e = ys - Xs @ b
    sg = np.column_stack([np.bincount(cl, weights=Xs[:, j] * e) for j in range(X.shape[1])])
    V = XtXi @ (sg.T @ sg) @ XtXi * (G / (G - 1)) * ((NT - 1) / (NT - K - 1))
    return b, np.sqrt(np.diag(V))

# (label, outcome, regressors, terms to test)
TESTS = [
 ("Table 5 col 1", "Bid1",  ["Cost", "Productive", "Round"], ["Cost", "Productive", "Round"]),
 ("Table 5 col 2", "Bid1",  ["Cost", "Productive", "Round", "Female", "GenderOther", "Risk"], ["Female", "GenderOther", "Risk"]),
 ("Table 5 col 3", "Bid2",  ["Cost", "Productive", "Round"], ["Cost", "Productive", "Round"]),
 ("Table 5 col 4 (losers, winner effects)", "Bid2", ["Cost", "Productive", "Wbase", "WcostE", "WprodE", "Round"],
     ["Cost", "Productive", "Wbase", "WcostE", "WprodE", "Round"]),
 ("Table 5 col 4 (winners' treatment effects)", "Bid2", ["CostL", "ProdL", "Winner", "CostW", "ProdW", "Round"], ["CostW", "ProdW"]),
 ("Table 5 col 4 (interactions)", "Bid2", ["Cost", "Productive", "Winner", "CxW", "PxW", "Round"], ["CxW", "PxW"]),
 ("Table 5 col 5 (losers, winner effects)", "Bid2", ["Cost", "Productive", "Wbase", "WcostE", "WprodE", "WinProb1", "Round"],
     ["Cost", "Productive", "Wbase", "WcostE", "WprodE", "WinProb1", "Round"]),
 ("Table 5 col 5 (winners' treatment effects)", "Bid2", ["CostL", "ProdL", "Winner", "CostW", "ProdW", "WinProb1", "Round"], ["CostW", "ProdW"]),
 ("Table 5 col 5 (interactions)", "Bid2", ["Cost", "Productive", "Winner", "CxW", "PxW", "WinProb1", "Round"], ["CxW", "PxW"]),
 ("Table 5 col 6", "Total", ["Cost", "Productive", "Round"], ["Cost", "Productive", "Round"]),
]
JOBS = [(lab, y, cols, term) for (lab, y, cols, terms) in TESTS for term in terms]

def exact_p(job):
    lab, yname, cols, term = job
    y = d[yname].to_numpy(float)
    X = design(cols)
    j = cols.index(term) + 1
    b, se = re_fit(y, X)
    t_obs = b[j] / se[j]
    Xr = np.delete(X, j, axis=1)
    br, _ = re_fit(y, Xr)
    yhat_r = Xr @ br
    er = y - yhat_r
    ts = np.empty(len(COMBOS))
    for k, c in enumerate(COMBOS):
        w = W6[c]
        bs, ses = re_fit(yhat_r + w[cl] * er, X)
        ts[k] = bs[j] / ses[j]
    p_exact = np.mean(np.abs(ts) >= abs(t_obs) - 1e-12)
    p_t = 2 * (1 - stats.t.cdf(abs(t_obs), df=G - 1))
    return lab, term, b[j], se[j], t_obs, p_t, p_exact

if __name__ == "__main__":
    with Pool(2) as pool, open(OUT, "w") as f:
        for lab, term, b, se, t, pt, pe in pool.imap(exact_p, JOBS):
            line = f"{lab:45s} | {term:12s} | coef {b:9.3f} | SE {se:7.3f} | t {t:7.3f} | clustered-t p {pt:.3f} | bootstrap p {pe:.5f}"
            print(line, flush=True); f.write(line + "\n")
    print("done", flush=True)

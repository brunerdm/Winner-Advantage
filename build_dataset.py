"""Build the participant-round analysis file from the raw oTree exports.

Usage: python3 build_dataset.py RAW_DIR OUT_CSV
RAW_DIR contains the six treatment folders (Baseline_*, Cost_*, Productivity_*).
Output: one row per participant per round (after dropping demo rows).
"""
import glob, os, sys
import numpy as np
import pandas as pd

RAW = sys.argv[1]
OUT = sys.argv[2]

# session order used in the paper: 1-2 BASELINE, 3-4 COST, 5-6 PRODUCTIVE
SESSIONS = [
    ("Baseline_04302024_1400",   1, 0, 0),
    ("Baseline_05032024_1000",   2, 0, 0),
    ("Cost_05022024_1200",       3, 1, 0),
    ("Cost_05032024_1200",       4, 1, 0),
    ("Productivity_04302024_1600", 5, 0, 1),
    ("Productivity_05022024_1400", 6, 0, 1),
]

frames = []
for folder, sess, cost, prod in SESSIONS:
    main = glob.glob(os.path.join(RAW, folder, "*.csv"))
    main = [f for f in main if os.path.basename(f).split("_")[0] in ("Control", "Cost", "Productivity")]
    assert len(main) == 1, (folder, main)
    r = pd.read_csv(main[0])
    r = r[r["session.is_demo"] == 0].copy()          # drop demo rows (Baseline 04-30 has 2 demo participants)
    r["session"] = sess
    r["Cost"] = cost
    r["Productive"] = prod
    r["PlayerID"] = r["participant.id_in_session"]
    r["Round"] = r["subsession.round_number"]
    r["GroupID"] = r["group.id_in_subsession"]
    r["Bid1"] = r["player.investment"]
    r["Bid2"] = r["player.investment_2"]
    r["Belief1"] = r["player.Perceived_investment"]
    r["Belief2"] = r["player.Perceived_investment_2"]
    r["Winner1"] = r["player.is_winner"]
    r["Winner2"] = r["player.is_winner_2"]
    # demographics and risk from the survey and risk exports, matched on participant ID
    sv = pd.read_csv(glob.glob(os.path.join(RAW, folder, "survey_*.csv"))[0])
    sv = sv[sv["session.is_demo"] == 0][["participant.id_in_session", "player.gender"]]
    rk = pd.read_csv(glob.glob(os.path.join(RAW, folder, "risk_*.csv"))[0])
    rk = rk[rk["session.is_demo"] == 0][["participant.id_in_session", "player.risk"]]
    dm = sv.merge(rk, on="participant.id_in_session")
    dm = dm.rename(columns={"participant.id_in_session": "PlayerID"})
    r = r.merge(dm, on="PlayerID", how="left")
    r["Female"] = (r["player.gender"] == "Woman").astype(int)
    r["GenderOther"] = r["player.gender"].isin(["Other", "Prefer not to say"]).astype(int)
    r["Risk"] = r["player.risk"]
    r["Earnings1"] = r["player.payoff1"]
    r["Earnings2"] = r["player.payoff2"]
    frames.append(r[["session", "Cost", "Productive", "PlayerID", "Round", "GroupID",
                     "Bid1", "Bid2", "Belief1", "Belief2", "Winner1", "Winner2",
                     "Female", "GenderOther", "Risk", "Earnings1", "Earnings2"]])

d = pd.concat(frames, ignore_index=True)
d["Adv"] = ((d["Cost"] + d["Productive"]) > 0).astype(int)
d["UID"] = d["session"] * 1000 + d["PlayerID"]   # unique participant ID (PlayerID restarts in each session)

# Opponent bids: groups have exactly two members, so the opponent's bid is group sum minus own bid
key = ["session", "Round", "GroupID"]
assert (d.groupby(key)["PlayerID"].transform("size") == 2).all(), "groups must have two members"
d["opponent_bid1"] = d.groupby(key)["Bid1"].transform("sum") - d["Bid1"]
d["opponent_bid2"] = d.groupby(key)["Bid2"].transform("sum") - d["Bid2"]
mx = d.groupby(key)["Bid1"].transform("max")
d["HighBid1"] = (d["Bid1"] == mx).astype(float)
tie = d.groupby(key)["Bid1"].transform("nunique") == 1
d.loc[tie, "HighBid1"] = np.nan
d.to_csv(OUT, index=False)
print("rows", len(d), "participants per session", d.groupby("session")["PlayerID"].nunique().to_dict())

# ---------------------------------------------------------------------------
# Best responses (integer grid 0..200). Prize V = 200.
# In the advantage treatments the first-contest winner's effective investment is
# 8 x bid in PRODUCTIVE, and the loser's is bid / 8 in COST (raw columns
# group.effective_investment_*_2). Both are the same as multiplying the winner's
# bid by 8 relative to the loser's, so one multiplier is used: 8 for the
# first-contest winner in advantage sessions, 1 otherwise.
# Payoff from bid b against opponent bid o: V * P - b, with
# P = m_own*b / (m_own*b + m_opp*o), and P = 0 when both effective bids are zero.
# Ties: smallest bid.
# ---------------------------------------------------------------------------
V = 200.0
MULT = 8.0
grid = np.arange(0, 201, dtype=float)

def best_response(opp, m_own, m_opp):
    opp = np.asarray(opp, dtype=float)
    own_eff = grid[:, None] * m_own[None, :]
    opp_eff = opp[None, :] * m_opp[None, :]
    denom = own_eff + opp_eff
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(denom > 0, own_eff / denom, 0.0)
    payoff = V * p - grid[:, None]
    return grid[np.argmax(payoff, axis=0)]

m_own = np.where((d["Winner1"] == 1) & (d["Cost"] + d["Productive"] > 0), MULT, 1.0)
m_opp = np.where((d["Winner1"] == 0) & (d["Cost"] + d["Productive"] > 0), MULT, 1.0)
d["BR2"] = best_response(d["opponent_bid2"].values, m_own, m_opp)
d["BR2_belief"] = best_response(d["Belief2"].values, m_own, m_opp)
d.to_csv(OUT, index=False)
print("best responses written")

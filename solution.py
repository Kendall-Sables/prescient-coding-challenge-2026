"""
Prescient Coding Challenge 2026 -- your submission.

THIS IS THE ONLY FILE YOU MAY CHANGE.

You implement one function. The harness calls it once per trading day and hands
you a `hist` object holding every observation STRICTLY BEFORE that day. You
return the weights you want to hold for that day.

    generate_weights(hist, prev_weights, params) -> weights

What you get
------------
hist.date                 the day you are allocating for (no data for it yet)
hist.returns              DataFrame [date x asset] of daily returns, decimals
hist.prices               DataFrame [date x asset] of total-return index levels
hist.macro                DataFrame [date x macro feature]
hist.assets               list of the six asset codes, in order
hist.benchmark            Series of benchmark weights
hist.active_weight(w)     total active weight of w -- the number rule 3 tests

prev_weights              what you held yesterday. Trading away from it costs
                          money, so look at it.
params                    the PARAMS dict below, passed straight through

Optional extras, in case you want them: hist.cov() gives an EWMA covariance
matrix and hist.te(w) an ex-ante tracking error. No rule depends on either.

What you must return
--------------------
Six weights (dict, Series or array in hist.assets order) that sum to 1, are all
non-negative, sit within 10% of their benchmark weight, have a total active
weight of no more than 40%, keep total equity at or below 75% and gold at or
below 10%. `make_legal()` below
already does all of that -- you can leave it alone.

Declare every tuneable number in PARAMS. Parameter count is part of the score.

Run `python harness.py` to test on the practice window (calendar 2025), then
`python validate.py` before you submit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Every tuneable number lives here. Fewer is better.
# --------------------------------------------------------------------------- #

# A (Kendall) owns SIG_PARAMS. Every key starts with "sig_".
SIG_PARAMS = {
    "sig_growth":     1.0,   # overweight on each of the two equity legs
    "sig_income":     2.0,   # underweight on bonds and on cash
    "sig_gold":       2.0,   # overweight on gold, the only real diversifier
    "sig_yield_days": 750,   # lookback for "are SA bonds historically cheap?"
}

# B (Akhona) owns EXE_PARAMS. Every key starts with "exe_".
EXE_PARAMS = {
    "exe_tilt":  0.06,        # how far a 1-sigma signal moves a weight
    "exe_speed": 0.10,        # fraction of the gap to yesterday we close per day
}

# FROZEN. The harness reads len(PARAMS). Neither of us edits this line.
PARAMS = {**SIG_PARAMS, **EXE_PARAMS}

# The rules, restated locally so this file reads on its own.
ACTIVE_BAND = 0.10       # per asset, distance from benchmark
ACTIVE_BUDGET = 0.40     # total, summed over assets
EQUITY = ["SA_EQUITY", "GLOBAL_EQUITY"]
EQUITY_CAP = 0.75        # total equity, whatever the bands allow
GOLD_CAP = 0.10


# --------------------------------------------------------------------------- #
# <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>
#
# This is your playground. Delete or rewrite anything here. What follows is a
# deliberately naive starting point so you can see the shape of a working
# answer. It is NOT a good answer -- on the practice window it loses to the
# benchmark. Your job is to do better.
#
# Three steps:
#   1. build a signal (here: a plain inverse-volatility tilt, which knows
#      nothing at all about expected return),
#   2. make the weights legal,
#   3. move only part of the way from yesterday, so you do not pay the full
#      trading cost every day.
#
# Steps 2 and 3 are plumbing. Keep them. Step 1 is the actual question, and
# inverse volatility is a poor answer to it: it will always prefer cash and
# bonds, whatever is happening in the world.
#
# Things worth thinking about. Which of these six assets actually diversifies
# the other five? Gold and global equity are both priced in rands -- what does
# that mean when the currency moves? The macro file has a term spread and a
# policy rate in it; what should a steepening curve do to your bond weight? And
# look at the cost table in the README before you trade property daily.
# --------------------------------------------------------------------------- #


def build_signal(hist, params) -> pd.Series:
    """Score per asset. Positive means overweight, negative means underweight.

    One structural view, plus one valuation check on the leg most likely to be
    wrong.

    The view. The benchmark holds 32.5% in bonds and cash -- the two assets that
    have lagged it in most calendar years -- and 2.5% in gold, the only asset
    here that is negatively correlated with SA equity, bonds and property. So
    fund the equities and gold out of the income block. Gold runs all the way to
    its 10% cap, because a cap is the only good reason not to hold more of the
    one thing that actually diversifies. Property gets no view: it is the most
    expensive line to trade at 35bps and the one with the worst tail.

    The check. A high bond yield is a high expected return, so the bond
    underweight is the leg that valuation can veto. When the SA 10-year sits a
    standard deviation or more above its own three-year average, the underweight
    fades to nothing. It only ever shrinks the position, never reverses it --
    this is position sizing, not a call on where yields go next.
    """
    growth = float(params["sig_growth"])
    income = float(params["sig_income"])
    gold = float(params["sig_gold"])

    # how cheap are SA bonds against their own recent history, in sigmas
    yields = hist.macro["sa_10y"].tail(int(params["sig_yield_days"])).dropna()
    cheap = 0.0
    if len(yields) >= 250 and yields.std() > 0:
        cheap = float((yields.iloc[-1] - yields.mean()) / yields.std())
    bond_conviction = min(max(1.0 - cheap, 0.0), 1.0)

    view = {
        "SA_EQUITY":     growth,
        "GLOBAL_EQUITY": growth,
        "SA_BONDS":     -income * bond_conviction,
        "SA_CASH":      -income,
        "SA_PROPERTY":   0.0,
        "GOLD":          gold,
    }
    return pd.Series({a: view.get(a, 0.0) for a in hist.assets}, dtype=float)


def make_legal(weights: pd.Series, hist) -> pd.Series:
    """Force `weights` to satisfy every rule. You can leave this alone.

    Everything happens in active space -- how far each asset sits from its
    benchmark weight -- because that is how the rules are written.

    The loop is there because the steps interfere: forcing the active weights
    to net to zero (so the portfolio sums to 1) can push an asset back outside
    its band. A few passes settles it. The budget scaling goes last and is safe
    there: shrinking every active weight toward zero cannot breach a band, a
    cap, or non-negativity.
    """
    bm = hist.benchmark
    active = weights.reindex(hist.assets).astype(float) - bm

    for _ in range(50):
        active = active.clip(lower=-ACTIVE_BAND, upper=ACTIVE_BAND)  # rule 2
        active = active.clip(lower=-bm)                              # keeps weights >= 0
        # rule 4: total equity cap. Trim the equity block back, sharing the
        # cut over whichever equity assets still have room to come down.
        eq_excess = (bm[EQUITY] + active[EQUITY]).sum() - EQUITY_CAP
        eq_full = eq_excess > -1e-12
        if eq_excess > 0:
            floor = np.maximum(-ACTIVE_BAND, -bm[EQUITY])
            down = (active[EQUITY] - floor).clip(lower=0)
            if down.sum() > 1e-15:
                active[EQUITY] = active[EQUITY] - eq_excess * down / down.sum()

        active["GOLD"] = min(active["GOLD"], GOLD_CAP - bm["GOLD"])  # rule 5

        excess = active.sum()          # must be zero for weights to sum to 1
        if abs(excess) < 1e-12:
            break
        # give the correction to the assets that have room to absorb it
        room = (ACTIVE_BAND - active) if excess < 0 else (active + bm).clip(lower=0)
        room = room.clip(lower=0)
        if excess < 0 and eq_full:
            room[EQUITY] = 0.0   # equity is at its cap -- top up elsewhere
        if room.sum() <= 1e-15:
            break
        active = active - excess * room / room.sum()

    total = active.abs().sum()                                       # rule 3
    if total > ACTIVE_BUDGET:
        active = active * (ACTIVE_BUDGET / total)

    return bm + active


def generate_weights(hist, prev_weights, params):
    """Return the six portfolio weights to hold on hist.date."""
    bm = hist.benchmark

    # not enough history to estimate anything: sit on the benchmark
    if len(hist.returns) < 260:
        return bm.to_dict()

    # 1. signal -> target weights around the benchmark
    signal = build_signal(hist, params)
    target = make_legal(bm + float(params["exe_tilt"]) * signal, hist)

    # 2. trade gradually toward the target rather than jumping to it
    prev = prev_weights.reindex(hist.assets)
    w = prev + float(params["exe_speed"]) * (target - prev)

    return make_legal(w, hist).to_dict()


# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>

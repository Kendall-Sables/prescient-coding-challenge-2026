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

# Opinion: what we think, independent of what we hold or what trading costs.
SIG_PARAMS = {
    "sig_rand_days":  60,     # trailing window for the rand-axis move
    "sig_vol_days":  250,     # lookback for normalising that move to a z-score
    "sig_damp_z":    1.5,     # z at which conviction peaks, then decays
}

# Execution: how hard we act on that opinion, and how fast we get there.
EXE_PARAMS = {
    "exe_tilt_size": 0.10,    # how far a 1-sigma view moves a weight
    "exe_band_mult": 10.0,    # no-trade half-width, as a multiple of asset cost
}

# The harness scores len(PARAMS). The "sig_"/"exe_" prefixes keep this merge
# collision-free -- a duplicate key would silently drop a value and undercount.
PARAMS = {**SIG_PARAMS, **EXE_PARAMS}

# The rules, restated locally so this file reads on its own.
ACTIVE_BAND = 0.10       # per asset, distance from benchmark
ACTIVE_BUDGET = 0.40     # total, summed over assets
EQUITY = ["SA_EQUITY", "GLOBAL_EQUITY"]
EQUITY_CAP = 0.75        # total equity, whatever the bands allow
GOLD_CAP = 0.10

# One-way trading cost per asset, as a decimal. A published rule of the game,
# restated here like the bands above -- not market data.
COSTS = pd.Series({
    "SA_EQUITY":     0.0015,
    "GLOBAL_EQUITY": 0.0020,
    "SA_BONDS":      0.0008,
    "SA_CASH":       0.0001,
    "SA_PROPERTY":   0.0035,
    "GOLD":          0.0025,
})

# Loading on the offshore-vs-local axis. Signs are each asset's beta to the rand;
# magnitudes are ROUNDED on purpose -- those betas drift hard through time (SA
# equity ran -0.43 in 2008 and +0.15 in 2013), so a precise estimate would be
# false precision. They sum to zero, so the tilt funds itself offshore-against-
# local rather than out of cash, which is what the spread is actually made of.
RAND_LOADING = pd.Series({
    "GOLD":           1.0,   # strongest rand hedge
    "GLOBAL_EQUITY":  1.0,
    "SA_CASH":        0.0,   # neutral ballast, beta 0.00
    "SA_EQUITY":     -1.0,   # strongest local
    "SA_BONDS":      -0.5,
    "SA_PROPERTY":   -0.5,
})


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


# SCALE CONTRACT for build_signal(). The number returned per asset is a view in
# standard-deviation units, NOT a weight:
#     ~0.0  no view        +/-1.0  a normal view        +/-3.0  rare, extreme
# size_and_trade() owns the translation from view into rands. Nothing in here
# knows about weights, trading costs, or what we held yesterday.
def build_signal(hist, params) -> pd.Series:
    """Score per asset: the rand axis, played for reversion.

    One axis explains most of the cross-section here -- offshore versus local,
    driven by the rand. Gold and global equity are the hedges, the three SA
    assets move against it, cash is ballast.

    That axis reverts rather than trends. A rand that has ALREADY weakened over
    60 days tends to give some of it back, and giving it back hurts the offshore
    assets -- so the tilt is the NEGATIVE of trailing rand momentum.

    Pure opinion: no trading logic, no cost logic, never reads prev_weights.

    hist.macro ends at t-1 by construction, so this window cannot see the day it
    allocates for. That lag is the difference between this signal and its own
    inverse: unlagged, the same test reads +13.5 instead of -4.1.
    """
    lb = int(params["sig_rand_days"])
    logzar = np.log(hist.macro["usdzar"])
    if len(logzar) <= lb:
        return pd.Series(0.0, index=hist.assets)

    # the 60-day move, in units of its own volatility -> a z-score
    mom = float(logzar.iloc[-1] - logzar.iloc[-1 - lb])
    sigma = float(logzar.diff().tail(int(params["sig_vol_days"])).std() * np.sqrt(lb))
    if not sigma > 0:
        return pd.Series(0.0, index=hist.assets)
    z = float(np.clip(mom / sigma, -6.0, 6.0))   # numerical bound, not a view

    # Damped response: lean into a normal dislocation, but back off when the move
    # is so large it looks like a regime rather than a stretch. Fading a one-way
    # collapse is how a reversion signal dies -- 2008, 2013 and 2018 are the
    # biggest one-way rand years in development, and all three invert.
    z0 = float(params["sig_damp_z"])
    response = z * np.exp(-0.5 * (z / z0) ** 2)

    return -response * RAND_LOADING.reindex(hist.assets)


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


def size_and_trade(signal, hist, prev_weights, params) -> pd.Series:
    """Turn a view into the weights we actually hold today.

    Owns everything build_signal() is not allowed to know: how much a view is
    worth in rands, what we held yesterday, and what it costs to move. Always
    ends by calling make_legal(), so callers never think about the rules.
    """
    bm = hist.benchmark
    prev = prev_weights.reindex(hist.assets)

    # 1. view -> active weights, with SA_CASH as the funding leg. Cash costs 1bp
    #    to trade and has no beta to the rand axis, so absorbing the residual
    #    there is both cheap and neutral. Netting to zero also leaves make_legal()
    #    nothing to route, which stops it quietly funding our positions out of
    #    whichever asset happens to have room.
    active = float(params["exe_tilt_size"]) * signal.reindex(hist.assets).fillna(0.0)
    active["SA_CASH"] = -active.drop("SA_CASH").sum()
    target = make_legal(bm + active, hist)

    # 2. no-trade band per asset, scaled by that asset's own cost. Do nothing
    #    inside it; outside it, trade back to the EDGE rather than to the target,
    #    so we only ever pay for the part of the gap worth paying for.
    band = float(params["exe_band_mult"]) * COSTS.reindex(hist.assets)
    gap = target - prev
    move = np.sign(gap) * (gap.abs() - band).clip(lower=0.0)

    return make_legal(prev + move, hist)


def generate_weights(hist, prev_weights, params):
    """Return the six portfolio weights to hold on hist.date. FROZEN GLUE."""
    if len(hist.returns) < 260:        # too little history to estimate anything
        return hist.benchmark.to_dict()
    signal = build_signal(hist, params)
    return size_and_trade(signal, hist, prev_weights, params).to_dict()


# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>

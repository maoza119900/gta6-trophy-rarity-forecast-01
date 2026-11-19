"""gta6-trophy-rarity-forecast-01 entry: forecast each GTA VI trophy's Exophase earn rate
at the case target as five quantiles, by separating the two forces that move a rate.

A trophy's rate is holders / players_tracked. Two forces move it in opposite directions:
  * dilution -- new players join holding ~1 trophy each (the universal 100% one), so every
    other rate falls when Exophase refreshes (launch night: one refresh took tracked players
    638 -> 3,868 and cut a typical rate to ~0.17x-0.22x of its old value);
  * progression -- existing players earn more trophies over time, so holders grow.
A trend fitted to the raw rate conflates the two, and over-extrapolates the decline once
player inflow decelerates. So this program fits the trend to the dilution-free holders
series a_ij = H_ij / N_last (holders as a fraction of the checkpoint population), forecasts
the player count separately, and divides: rate(target) = a(target) * N_last / N(target).
When the history is flat (no trend can be fitted, as on launch night) it falls back to a
dilution prior: a refresh before the target cuts a typical rate to ~0.21x, with a
logit spread of 0.45 calibrated on the reconstructed practice answers (SPEC.md 5).

Python 3.13, standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped beside
# this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- flat-history (dilution prior) regime -------------------------------------
# Calibrated on the practice answers as reconstructed from the kit's documented
# aggregates (04:30Z read: N=3,868, 4,276 trophies earned; see SPEC.md section 5):
# a grid search over the four reconstruction scenarios puts the plateau at
# factor 0.20-0.22, sigma 0.35-0.5; 0.45 keeps a margin for error in how the 103
# extra holdings spread, instead of sitting at the sharpest point of the grid.
FLAT_FACTOR = 0.21        # median rate after a refresh, as a fraction of the last rate
FLAT_SIGMA = 0.45         # logit spread: covers the extras-distribution scenarios
FLAT_EARLY_BOOST = 1.15   # earliest-story trophies catch a share of the new players' extras
FLAT_EARLY_IDS = 14       # trophies 3..14 are the early story (id 2 is the universal one)

# --- trend regime ---------------------------------------------------------------
HALF_LIFE_H = 24.0        # recency weight half-life for the trend fit
SLOPE_CAP = 4.0           # |slope| cap, logit per log-hour (real slopes run ~2.4)
DELTA_CAP = 3.0           # |predicted logit move from last read| cap
N_GROWTH_CAP = 3.0        # N(target) at most this multiple of N(last)
N_FIT_WINDOW_H = 24.0     # player-count trend fitted on recent reads only: the
                          # inflow slope falls as the launch spike fades, so old reads
                          # would overstate N(target)
SIGMA_FLOOR = 0.35
SIGMA_CAP = 1.5
REFRESH_VAR = 0.15        # extra variance for refresh jumps between checkpoint and target
BASE_VAR = 0.04


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round(x * 10000) / 10000)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, s2, se_b2, xbar)."""
    sw = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return ybar, 0.0, 0.0, 0.0, xbar
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    n = len(xs)
    if n > 2:
        s2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
        s2 *= n / (n - 2.0)
    else:
        s2 = 0.0
    se_b2 = s2 / sxx if sxx > 0 else 0.0
    return a, b, s2, se_b2, xbar


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    target = parse_time(case["target"])
    t_target = max(0.0, (target - RELEASE).total_seconds() / 3600.0)
    x_target = math.log(max(0.05, 1.0 + t_target))

    # Per-read: time (hours since release), players tracked, rates.
    reads = []
    for h in history:
        t = (parse_time(h["at"]) - RELEASE).total_seconds() / 3600.0
        reads.append((t, h["players_tracked"], h["rates"]))
    t_last = reads[-1][0]
    n_last = reads[-1][1]
    last_rates = reads[-1][2]

    # Is the history (near-)flat? Count distinct (players, rates) snapshots.
    distinct = set()
    for _, n, rates in reads:
        distinct.add((n, tuple(r if r is None else round(r, 4) for r in rates)))
    flat = len(distinct) < 3

    # --- player-count forecast (trend regime only; flat history => N stays) ----
    # Fitted on recent reads only: the inflow slope falls as the launch spike fades.
    n_target = n_last
    if not flat:
        xs, ys, ws = [], [], []
        for t, n, _ in reads:
            if n and n > 0 and t >= t_last - N_FIT_WINDOW_H:
                xs.append(math.log(max(0.05, 1.0 + t)))
                ys.append(math.log(n))
                ws.append(0.5 ** ((t_last - t) / HALF_LIFE_H))
        if len(xs) >= 3:
            a_n, b_n, _, _, _ = wls(xs, ys, ws)
            b_n = max(-0.5, min(1.5, b_n))
            n_target = math.exp(a_n + b_n * x_target)
            n_target = min(max(n_target, float(n_last)), N_GROWTH_CAP * n_last)

    # --- per-trophy trend fits --------------------------------------------------
    # First pass: one weighted fit per trophy; the slopes are then shrunk toward
    # empirical-Bayes tier priors (see below).
    fits = []
    for i in range(n_tr):
        xs, ys, ws = [], [], []
        for t, n, rates in reads:
            r = rates[i]
            if r is None or n is None or n <= 0:
                continue
            holders = r * n / 100.0
            a_frac = holders / n_last if n_last > 0 else 0.0
            a_frac = min(0.9999, max(0.0001, a_frac))
            xs.append(math.log(max(0.05, 1.0 + t)))
            ys.append(logit(a_frac * 100.0))
            ws.append(0.5 ** ((t_last - t) / HALF_LIFE_H))
        if len(xs) >= 3:
            a, b, s2, se_b2, xbar = wls(xs, ys, ws)
            fits.append((a, b, s2, se_b2, xbar, ys[-1], xs[-1]))
        else:
            fits.append(None)
    # Empirical-Bayes shrinkage of slopes: with many reads a trophy's own slope is
    # precise (small se) and barely shrunk; with few reads it leans on its tier's
    # mean slope. A fixed shrink factor and a hard cap below the real slopes (~2.4)
    # both wrecked the extrapolation, so neither is used any more.
    def _eb_prior(idxs):
        bs = [fits[i][1] for i in idxs if fits[i] is not None]
        ses = [fits[i][3] for i in idxs if fits[i] is not None]
        if not bs:
            return 0.0, 0.0
        prec = [1.0 / (se + 1e-9) for se in ses]
        mean = sum(p * b for p, b in zip(prec, bs)) / sum(prec)
        var = sum(p * (b - mean) ** 2 for p, b in zip(prec, bs)) / sum(prec)
        tau2 = max(0.0, var - sum(ses) / len(ses))
        return mean, tau2

    all_idx = list(range(n_tr))
    global_prior, global_tau2 = _eb_prior(all_idx)
    tier_priors = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        idxs = [i for i, tr in enumerate(trophies) if tr["tier"] == tier]
        fitted = [i for i in idxs if fits[i] is not None]
        if len(fitted) >= 3:
            tier_priors[tier] = _eb_prior(idxs)
        else:
            tier_priors[tier] = (global_prior, global_tau2)

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            rate = 0.0
        if flat:
            # Dilution prior: a refresh before the target cuts the rate.
            if rate >= 99.99:
                median = 99.99
                sigma = 0.05
            else:
                factor = FLAT_FACTOR
                if 3 <= trophy["id"] <= FLAT_EARLY_IDS:
                    factor *= FLAT_EARLY_BOOST
                median = clamp(rate * factor)
                sigma = FLAT_SIGMA
        else:
            f = fits[i]
            prior, tau2 = tier_priors.get(trophy["tier"], (global_prior, global_tau2))
            if f is None:
                median = clamp(rate)
                sigma = SIGMA_FLOOR + 0.5
            else:
                a, b, s2, se_b2, xbar, y_last, x_last = f
                # Shrink toward the tier prior only as much as the slope's own
                # noise warrants; then extrapolate ANCHORED at the last observed
                # level (extrapolating from the fitted intercept instead silently
                # re-based the forecast whenever the cap or the shrink moved the
                # slope -- the bug that halved the synthetic-final medians).
                w = tau2 / (tau2 + se_b2) if tau2 > 0 else 0.0
                b = w * b + (1.0 - w) * prior
                b = max(-SLOPE_CAP, min(SLOPE_CAP, b))
                y_pred = y_last + max(-DELTA_CAP,
                                      min(DELTA_CAP, b * (x_target - x_last)))
                a_pred = min(0.99, max(0.0001, inv_logit_pct(y_pred) / 100.0))
                median = clamp(a_pred * n_last / n_target * 100.0)
                var = (max(s2, 0.02) + se_b2 * (x_target - x_last) ** 2
                       + REFRESH_VAR + BASE_VAR)
                sigma = math.sqrt(var)
                sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
            if rate >= 99.99:
                median = 99.99
                sigma = min(sigma, 0.1)
        lg = logit(median)
        qs = [round4(clamp(inv_logit_pct(lg + z * sigma))) for z in Z]
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

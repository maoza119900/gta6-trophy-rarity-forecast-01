# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/maoza119900/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/maoza119900/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one, from Exophase's
trophy page as the house recorded it. For every trophy it outputs five quantiles
(5/25/50/75/95%) of its earn rate at the case target, scored by pinball loss on
the logit scale.

## The model

A trophy's rate is holders / players tracked, and two forces move it in opposite
directions: **dilution** (new players join holding ~1 trophy each, so every other
rate falls when Exophase refreshes — launch night one refresh cut a typical rate
to ~0.2x) and **progression** (existing players earn more trophies, so holders
grow). A trend fitted to the raw rate conflates the two. This program fits the
trend to the dilution-free holders series `a = H / N_last`, forecasts the player
count separately from recent reads, and divides: `rate(target) =
a(target) * N_last / N(target)`. Slopes get empirical-Bayes shrinkage toward
their tier's mean, and the forecast is extrapolated anchored at the last observed
level. When the history is flat (launch night: no trend can be fitted) it falls
back to a dilution prior: a refresh before the target cuts a typical rate to
~0.21x, spread 0.45 logit, calibrated on the practice answers as reconstructed
from the kit's documented aggregates. See `SPEC.md` for the full write-up and
validation numbers.

## Run it

```sh
python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (or `/input`) and writes one
`<id>.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (or `/output`). Python
3.13, standard library only, no network, deterministic (two runs on the same
case write byte-identical output), ~0.05 s per practice case, well under the
8 s cap.

## Validation (local; there is no grade-local for this competition)

- Practice p00/p01, answers reconstructed from the kit's documented aggregates
  (04:30Z read: 3,868 players, 4,276 trophies earned): pinball loss **0.105–0.150**
  across four reconstruction scenarios, vs the starter baseline's 1.38–1.69
  (house documents ~1.51). Typical miss 0.02–0.05 pp, 90% coverage 0.98.
- Synthetic final (mechanistic simulator: refresh jumps, decelerating player
  inflow, per-trophy progression curves; three progression settings): loss
  **0.137–0.146**, vs persistence 0.50–0.60 and the house curve baseline
  0.22–0.27.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

# SPEC — gta6-trophy-rarity-forecast-01: Forecast GTA VI trophy earn rates at week one

Competition: `gta6-trophy-rarity-forecast-01` · $250 USDC, 20 places · opens 2026-11-19T11:30:00Z ·
deadline **2026-11-21T11:30:00Z (2026-11-21 19:30 CST)** · target 2026-11-26T00:00:00Z.
Account: make.fun `@vivid-tapir-15`, agent wallet `29fbnw6F6WBLrzSg8r82CKgNGkcY8Ao5bFD52mjZcaCL`.
GitHub owner for this entry: **maoza119900**. Kit:
`~/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/gta6-trophy-rarity-forecast-01-kit/`.
Requirements: `~/workspace/makefun/other-bounties/gta6-trophy-rarity-forecast-01.json`.
Work dir: `~/workspace/makefun/gta6-trophy-rarity-forecast-01/` (`bot/` is the repo).

## 1. The task

Forecast, for each of GTA VI's 55 PS5 trophies, its earn rate on Exophase at 2026-11-26T00:00:00Z
(one week after launch) as five quantiles (5/25/50/75/95%), from every read of the game's trophy
page the house recorded before the case checkpoint. Scored by mean pinball loss on the **logit**
scale (rates clamped to [0.01, 99.99] before logit), lower is better. A prize needs a loss
strictly below the best house entry (persistence baseline, fitted-curve baseline, house AI model).
Practice cases p00/p01: checkpoints 2026-11-19 03:00/03:30Z, 90-min horizon, answers are the
04:30/05:00Z reads. Final case f00: checkpoint at the deadline, horizon 6,510 min.

## 2. What the data actually does (read from the kit, not assumed)

- Exophase does **not** refresh continuously. Launch night the page was frozen from 23:48Z to
  04:00Z (638 players tracked, every rate identical), then one refresh between 04:00Z and 04:30Z:
  3,868 players, 4,276 trophies earned, every rate but the 100% one fell. The 05:00Z read was
  unchanged again. So a read shows rates as of Exophase's last refresh, hours apart.
- The aggregate identity holds exactly (verified on p00): `trophies_earned = Σ rate_i × N / 100`
  (Σrates = 147.78 → 942.8 ≈ 943). So holders per trophy are recoverable as integers:
  `H_i = round(rate_i × N / 100)`.
- The refresh is mostly **dilution**: 3,230 new players joined, holding 3,333 trophies between
  them — 3,230 of those are the universal 100% trophy ("Just Like Paradise", held by every
  tracked player), leaving only **103 extra holdings** spread over the other 54 trophies. So
  after a refresh, a typical trophy's rate is roughly its old rate × 638/3868 ≈ **0.165×**,
  plus a small extra for the earliest trophies. The persistence baseline (median = last rate,
  spread 0.5·√d) scores ≈1.51 on the practice for exactly this reason: its narrow band misses
  a ~1.8-logit downward jump on nearly every trophy (coverage 0).
- The two forces on a rate are opposed: **dilution** (new players join with ~1 trophy each,
  pushing rates down) and **progression** (existing players earn more trophies, pushing rates up).
  Any model that extrapolates the net rate trend conflates the two; if player inflow decelerates
  after launch (it must), the net trend over-extrapolates the decline.

## 3. The model (this entry, `main.py`, Python 3.13 stdlib only)

Work per trophy in logit space, but **decompose holders from dilution**:

1. From each history read j, recover `N_j` (players tracked) and holders
   `H_ij = rate_ij × N_j / 100` (rates are 2-decimal prints; H is near-integer).
2. **Dilution-free holders series.** Let `N_last` be the last read's player count and define
   `a_ij = H_ij / N_last`: the holders of trophy i as a fraction of the checkpoint population.
   `a` rises only when players actually earn the trophy — dilution is divided out.
3. **Trend in `a`.** Per trophy, weighted least squares of `logit(a_ij)` on
   `x_j = ln(1 + t_j)`, `t_j` = hours since release (2026-11-19T00:00Z), weights decaying
   with a ~24 h half-life (recent reads count most; all history is used, not just 24 h).
   Slopes get empirical-Bayes shrinkage toward their tier's mean slope (only as much as
   each slope's own noise warrants -- a fixed shrink factor wrecked the extrapolation),
   and the forecast is extrapolated ANCHORED at the last observed level, never from the
   fitted intercept.
4. **Player-count forecast.** Fit `ln N_j` on `ln(1 + t_j)` over recent reads, extrapolate to
   the target; clamp `N_target` to `[N_last, 4 × N_last]` (tracked players never fall).
5. **Median.** `rate_i(target) = a_i(target) × N_last / N_target`, with `a_i(target)` clamped
   to [0.0001, 0.99]. This is the key edge over the house curve baseline: the trend is fitted
   on dilution-free holders, and dilution is applied separately from the forecasted slowdown
   of player inflow.
6. **Flat-history regime (the practice cases).** If the history has < 3 distinct reads or
   zero variance (launch night: 7–8 identical reads), no trend can be fitted. Fall back to a
   dilution prior: assume a refresh before the target multiplies tracked players by ~4–5×
   (the documented launch-night refresh was 638 → 3,868 ≈ 6.1× over ~4.5 h of accumulation;
   a 90-min window captures part of that accumulation, so the prior centres lower), i.e.
   median rate ≈ last rate × **0.21**, plus a small uplift for the earliest-story trophies
   (the 103 extra holdings go to trophies new players can earn in their first hours).
   The 100% trophy stays ~100% (every new player holds it on arrival). Spread σ = 0.45
   (see §5 for how it was calibrated).
7. **Spread (calibrated, not the baseline's 0.5·√d).** Per trophy,
   `σ_i² = residual_var_i + slope_var_i × (x_T − x̄)² + refresh_var + base_var`,
   floored at 0.35 and capped at 1.5 logit units; in the flat regime σ = 0.45, calibrated
   on the reconstructed practice answers (grid plateau 0.35–0.5; 0.45 keeps a margin for
   error in how the 103 extra holdings spread — see §5).
   Quantiles: `q_τ = inv_logit(logit(median) + z_τ × σ_i)`, `z = (−1.645, −0.674, 0, 0.674, 1.645)`,
   rounded to 4 decimals, clamped to [0.01, 99.99], non-decreasing by construction.
8. **Safety (never an invalid case).** Any failure of the fit for one trophy falls back to
   persistence for that trophy; empty/short history falls back to the flat regime; output is
   always written, always one row per trophy, always ≤ 65,536 bytes. Deterministic: no
   randomness, no wall-clock use, fixed iteration order.

## 4. Why this should beat the house entries

- Persistence baseline: median = last rate. Wrong direction after every refresh (rates fall on
  dilution) and far too narrow on a 90-min horizon (σ = 0.125). Practice loss ≈ 1.51.
- Curve baseline: same as persistence on the practice (flat history → flat line, no residual);
  on the final it extrapolates the *net* rate trend, which mixes dilution with progression and
  uses only the last 24 h. This entry separates the two forces and uses all history.
- House AI model: unknown, but a single-call median+spread per trophy with no access to the
  refresh mechanics; a mechanistic decomposition with calibrated spread is the best available
  counter. (No local grade exists for this competition — the server grades; see §5.)

## 5. Validation (local — no grade-local for this competition)

`grade-local` refuses: "scorer and reference data are not published in the kit, so it is graded
on the server only". The practice answers are not shipped either, so validation reconstructs
them from the **documented aggregates** in competition.yaml (04:30Z read: N = 3,868,
trophies earned = 4,276; 05:00Z read unchanged), which pin Σrate′ = 110.55 and the extra-holdings
budget (103) exactly, and evaluates pinball loss under four answer scenarios:

- **A — pure dilution**: extra holdings 0 (rate′ = rate × 0.16495). Inconsistent with the
  documented total (Σrate′ = 107.88 ≠ 110.55); kept as a lower bound.
- **B — proportional extras**: the 103 extras distributed ∝ current holders (new players who
  play a bit earn in proportion to earnability): rate′ = rate × 0.2206. Matches Σrate′ exactly.
- **C — concentrated extras**: extras ∝ holders among the most-held trophies only.
- **D — uniform extras**: 103/54 ≈ 1.9 extra holders per trophy.

The model must beat the starter baseline clearly under **all four** scenarios, not just one —
that is the acceptance test (plus: determinism byte-identical on re-run, runtime ≪ 8 s,
valid output shape/size). **Result: loss 0.105–0.150 across the four scenarios
(mean 0.127 over B/C/D at the chosen factor 0.21 / σ 0.45 / early boost 1.15), vs the
starter's 1.382–1.688** — about 8–9% of baseline, typical miss 0.02–0.05 pp, 90% coverage
0.98. Cross-check: the baseline's mean loss over the four scenarios is 1.498, against
~1.51 documented in competition.yaml, so the reconstruction is close to the true answers.
The grid search (factor 0.18–0.24 × σ 0.35–0.9 × boost 1.0–1.3) has its plateau at
σ 0.35–0.5; 0.45 was chosen off the sharpest point to keep a margin for error in how the
103 extra holdings spread.

A synthetic final-like history (mechanistic simulator: refresh jumps, player-inflow decay,
per-trophy progression curves; `sim.py`, fixed seeds) checks the trend regime against
the house curve baseline on a 4.5-day horizon, since the practice cannot exercise it.
**Result (loss): slow 0.146 / base 0.137 / fast 0.146, vs persistence 0.50–0.60 and the
house curve 0.22–0.27** — 54–65% of the house curve's loss in all three progression
settings. The first version of the trend regime scored 0.98–1.54 there (worse than
persistence): it extrapolated from the fitted intercept with a hard slope cap of 2.0 and
a fixed 0.7 shrink toward the tier mean, which halved the medians. Fixed by anchoring
the extrapolation at the last observed level, empirical-Bayes shrinkage, cap 4.0, and
fitting the player-count trend on the last 24 h only (the inflow slope falls as the
launch spike fades; fitting all history overstated N(target) 169k vs ~103k true).

## 6. Limits

- Python 3.13, stdlib only, no network. `python3 main.py`, reads `$MAKEFUN_INPUT_DIR` (else
  `/input`), writes `$MAKEFUN_OUTPUT_DIR` (else `/output`), one `<id>.out` per `*.in`.
- ≤ 8 s, 1 CPU, 1024 MB per case; two runs byte-identical.
- Repo: public GitHub, owner maoza119900, created after 2026-11-19T11:30:00Z, MIT LICENSE
  (first line `SPDX-License-Identifier: MIT`), `main.py` at root, `README.md`.
- This build only: no repo creation, no bundle, no submit, no prompt.txt (parent's steps).

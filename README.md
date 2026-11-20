# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rickyarmnd/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rickyarmnd/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for the make.fun bounty [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

For each PlayStation 5 trophy it forecasts the Exophase earn rate at 2026-11-26T00:00:00Z: five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the percent of tracked players holding the trophy.

## Method

The rate is `100 * earners / tracked`, so the forecast is really the ratio `earners / tracked`. It is projected in log space: `log(earners)` and `log(tracked)` are each fitted against time with non-negative slopes over the reads in the window before the checkpoint, and the two projections are differenced. A damped power law carries a fresh launch cadence forward when the history is flat, near-100% trophies are clamped to persistence, and the quantile spread grows like the square root of the horizon with a downward skew.

## Run

    node --experimental-strip-types main.ts

Reads the case file at `MAKEFUN_INPUT_DIR` (default `/input`) and writes the forecast to `MAKEFUN_OUTPUT_DIR` (default `/output`).

MIT licensed.

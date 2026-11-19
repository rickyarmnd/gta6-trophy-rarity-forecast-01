// gta6-trophy-rarity-forecast-01
//
// Forecasts each PlayStation 5 trophy's Exophase earn rate at a case's target.
//
// The rate Exophase prints is earners divided by players tracked, both of which move. On the logit
// scale that splits cleanly:
//
//     logit(rate) = log(earners) - log(tracked - earners)
//
// so a growing player base pushes every rate down by roughly the same amount, while a trophy whose
// earners keep arriving pushes its own rate back up. The house recording gives both effects: a read
// whose players_tracked rose while a rate stayed flat is a pure denominator move, and a rate that
// rose faster than the denominator is earners catching up.
//
// So each trophy is modelled as a drift on the logit scale, fitted from its own history and shrunk
// toward the drift every trophy shares, then projected over the horizon under a damped power law:
// player growth decays, so the fitted per-day drift is not carried forward whole. The spread around
// that projection widens with the square root of the horizon, and it is skewed downward because a
// refresh that outruns the forecast costs rare trophies more than it gains common ones.
//
// Everything here is derived from the case file alone: the program runs with no network.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6449, -0.6745, 0, 0.6745, 1.6449];

/** Rates are clamped before the logit: 0% and 100% have none. */
const CLAMP_LO = 0.01;
const CLAMP_HI = 99.99;

/** The spread every forecast keeps even over a horizon of minutes: refresh jitter. */
const SIGMA_BASE = 0.55;
/** How fast the spread grows with the horizon, in logit units per square-root day. */
const SIGMA_GROWTH = 0.9;
/** The widest spread a trophy is given, so one bad history cannot flatten the whole row. */
const SIGMA_MAX = 2.0;
/** Extra spread for trophies so rare that a single earner moves them a lot. */
const SIGMA_RARE = 0.35;
/** The rarity below which that extra spread applies, in percent. */
const RARE_BELOW = 0.75;

/** How much of a trophy's own fitted drift survives the shrink toward the pooled one. */
const OWN_WEIGHT = 0.6;
/** The exponent that damps a fitted per-day drift over the horizon: growth decays. */
const DAMPING = 0.55;
/** The drift is clipped: no trophy is projected to collapse or to explode. */
const DRIFT_MIN = -2.6;
const DRIFT_MAX = 0.6;
/** How far the median is pulled below the projection, in units of sigma. */
const SKEW = 0.28;
/** The shortest history, in days, that is worth fitting a per-trophy drift to. */
const MIN_SPAN_DAYS = 0.04;
/** The fewest reads a trophy needs before its own drift is trusted. */
const MIN_POINTS = 3;

interface Trophy {
  id: number;
  tier: string;
  points: number;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(CLAMP_HI, Math.max(CLAMP_LO, x));
const logit = (pct: number): number => {
  const p = clamp(pct) / 100;
  return Math.log(p / (1 - p));
};
const expitPercent = (x: number): number => 100 / (1 + Math.exp(-x));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

const minutesOf = (iso: string): number => Date.parse(iso) / 60_000;

/** The median of a list of numbers, or NaN when it is empty. */
function median(values: number[]): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** One trophy's history as (days since the first read, logit rate), nulls dropped. */
function seriesOf(history: Read[], index: number): Array<{ day: number; x: number }> {
  const points: Array<{ day: number; x: number }> = [];
  let origin: number | null = null;
  for (const read of history) {
    const rate = read.rates[index];
    if (rate === null || rate === undefined || !Number.isFinite(rate)) continue;
    const t = minutesOf(read.at);
    if (!Number.isFinite(t)) continue;
    if (origin === null) origin = t;
    points.push({ day: (t - origin) / 1440, x: logit(rate) });
  }
  return points;
}

/**
 * The trophy's own drift per day: the slope between its first and last reading. A straight
 * two-point slope beats a regression here because the histories are short and monotone, and the
 * endpoint carries the freshest state of the refresh cycle.
 */
function ownDrift(points: Array<{ day: number; x: number }>): number | null {
  if (points.length < MIN_POINTS) return null;
  const first = points[0];
  const last = points[points.length - 1];
  const span = last.day - first.day;
  if (!(span >= MIN_SPAN_DAYS)) return null;
  const slope = (last.x - first.x) / span;
  return Number.isFinite(slope) ? slope : null;
}

/** One trophy's row: five quantiles in percent, non-decreasing. */
function forecastTrophy(points: Array<{ day: number; x: number }>, pooled: number, horizonDays: number): number[] {
  const last = points.length > 0 ? points[points.length - 1].x : logit(1);
  const own = ownDrift(points);
  const drift = own === null ? pooled : OWN_WEIGHT * own + (1 - OWN_WEIGHT) * pooled;

  const projected = last + drift * Math.pow(horizonDays, DAMPING);
  const moved = Math.min(DRIFT_MAX, Math.max(DRIFT_MIN, projected - last)) + last;

  const rare = expitPercent(last) < RARE_BELOW ? SIGMA_RARE : 0;
  const sigma = Math.min(SIGMA_MAX, Math.sqrt(SIGMA_BASE ** 2 + SIGMA_GROWTH ** 2 * horizonDays) + rare);

  const row: number[] = [];
  for (let i = 0; i < LEVELS.length; i += 1) {
    const z = Z[i];
    // A downward skew: the low quantiles reach further than the high ones, because a refresh that
    // brings in new players costs the rare trophies more than a quiet period gains them.
    const skew = SKEW * sigma * (z < 0 ? z : 0);
    row.push(round4(expitPercent(moved + z * sigma + skew)));
  }
  for (let i = 1; i < row.length; i += 1) {
    if (row[i] < row[i - 1]) row[i] = row[i - 1];
  }
  return row;
}

/** Forecast one case, writing <id>.out to the output directory. */
function forecastCase(c: Case): void {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case ${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;

  // The drift every trophy shares, taken from the trophies that could be fitted at all: it is the
  // fallback for a trophy whose history is too short to say anything about itself.
  const fitted: number[] = [];
  const series = c.trophies.map((_trophy, index) => {
    const points = seriesOf(c.history, index);
    const own = ownDrift(points);
    if (own !== null) fitted.push(own);
    return points;
  });
  const pooled = fitted.length > 0 ? median(fitted) : 0;

  const rows = c.trophies.map((trophy, index) => ({ id: trophy.id, q: forecastTrophy(series[index], pooled, horizonDays) }));
  const out = { trophies: rows };
  writeFileSync(join(OUTPUT_DIR, `${c.id}.out`), JSON.stringify(out), 'utf8');
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((name) => name.endsWith('.in')).sort();
for (const file of inputs) {
  const parsed = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  forecastCase(parsed);
}

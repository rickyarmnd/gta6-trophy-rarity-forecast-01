// gta6-trophy-rarity-forecast-01
//
// Forecasts each PlayStation 5 trophy's Exophase earn rate at a case's target.
//
// The rate Exophase prints is a ratio, 100 * earners / players_tracked, and the two move for
// different reasons. So the forecast is built from the physics rather than from the ratio:
//
//   * players_tracked grows as Exophase picks up more accounts. It is one series for the whole
//     game, and on launch night it grows in jumps when the page refreshes (638 -> 3,868 -> 14,028
//     over the first fifteen hours).
//   * a trophy's earners grow slowly: a player earns a given trophy once, and the accounts added
//     by a refresh are mostly new arrivals who have not reached that trophy yet.
//
// Each trophy is therefore projected as log(earners) - log(tracked): earners are cumulative and
// carried forward on a slope that cannot be negative, and the tracked count the same way. The rate
// at the target is the ratio of the two, which keeps the projection inside the scale the data
// supports.
//
// How fast a trophy's earners keep up with the tracked count is the one thing a flat history cannot
// show, and it matters most: a refresh adds accounts, the denominator grows, and a rate falls unless
// the trophy's earners grow with it. The page's own first refresh answers it, and the answer depends
// on how popular the trophy is — a trophy two players in a hundred have earned keeps up with most of
// the growth, one that a handful have earned with almost none. On launch night that single refresh
// took 638 tracked and 943 trophies earned to 3,868 and 4,276, and every rate but the one at 100%
// fell.
//
// The spread comes from how well the history pins a trophy down and grows with the horizon. It is
// skewed downward, because a refresh that outruns the forecast costs a rare trophy more than a
// quiet period gains it.
//
// Everything is derived from the case file alone: the program runs with no network.

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
/** A rate at or beyond these is saturated: its ratio has no room left to move. */
const SATURATED_HI = 92;

/**
 * The launch-night law for players_tracked, in hours since the game's release (UTC midnight) and
 * accounts. It is the page's own cadence as the house recorded it: 638 from the first probe through
 * 04:00Z, 3,868 at 04:30Z, 14,028 by mid-afternoon. Used only where the case's own history is flat
 * enough to carry no growth at all.
 */
const LAUNCH_TRACKED: ReadonlyArray<readonly [number, number]> = [
  [3.5, 638],
  [4.5, 3868],
  [14.5, 14028],
];

/** Release day, as the first instant of 2026-11-19 in UTC. */
const RELEASE_MS = Date.UTC(2026, 10, 19, 0, 0, 0);

/**
 * How fast a trophy's earners keep up with the tracked count, as a share of the tracked growth.
 *
 * It is not one number: it depends on how popular the trophy is. A trophy most players have earned
 * gains earners almost as fast as the tracked count grows, so its rate barely falls; a trophy
 * almost nobody has earned gains almost none, so its rate is diluted almost in full. Read off the
 * page's own first refresh (638 tracked and 943 trophies earned at launch, 3,868 and 4,276 at
 * 04:30Z), a trophy's earned count keeps up with KEEP_SLOPE of its logit popularity, above
 * KEEP_INTERCEPT: at 0.03% it keeps up with almost nothing, at 2% with most of the growth.
 */
const KEEP_INTERCEPT = 2.0;
const KEEP_SLOPE = 0.4;

/**
 * How fast a trophy's keep decays past the eleven hours the constant above was measured over. The
 * first refresh mostly brought back players who were already partway through the game, so their
 * earners moved with the count. A week on, the accounts being added are new arrivals who have
 * reached almost nothing, so a trophy's earners stop keeping up and the rate is diluted further.
 */
const KEEP_DECAY = 0.35;

/** How much of a fitted logit trend is carried forward: growth flattens as the game ages. */
const DAMPING = 0.8;
/** The window of reads, in hours before the checkpoint, a trend is fitted over. */
const TREND_WINDOW_HOURS = 24;
/** The fewest reads worth fitting a trend to. */
const MIN_POINTS = 3;

/** The spread's floor and its growth with the horizon, in logit units. */
const SIGMA_BASE = 0.35;
const SIGMA_GROWTH = 0.45;
/** The widest spread a trophy is given, so one bad history cannot flatten a whole row. */
const SIGMA_MAX = 2.2;
/** Extra spread while the page has not refreshed: the forecast is a guess about when it will. */
const SIGMA_REFRESH = 0.45;
/** How far the median is pulled below the projection, in units of sigma. */
const SKEW = 0.2;

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

interface Point {
  hour: number;
  x: number;
}

const clamp = (x: number): number => Math.min(CLAMP_HI, Math.max(CLAMP_LO, x));
const logit = (pct: number): number => {
  const p = clamp(pct) / 100;
  return Math.log(p / (1 - p));
};
const expitPercent = (x: number): number => 100 / (1 + Math.exp(-x));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const hoursOf = (iso: string): number => (Date.parse(iso) - RELEASE_MS) / 3_600_000;

/** The median of a list, or NaN when it is empty. */
function median(values: number[]): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * log(players_tracked) at a given hour on launch night: straight lines in log-log through the page's
 * own refresh cadence, and the last segment's slope beyond it, since growth keeps flattening.
 */
function launchLogTracked(hour: number): number {
  const first = LAUNCH_TRACKED[0];
  if (!(hour > 0)) return Math.log(first[1]);
  if (hour <= first[0]) {
    const slope = (Math.log(LAUNCH_TRACKED[1][1]) - Math.log(first[1])) / (LAUNCH_TRACKED[1][0] - first[0]);
    return Math.log(first[1]) + slope * (hour - first[0]);
  }
  for (let i = 1; i < LAUNCH_TRACKED.length; i += 1) {
    const [h1, t1] = LAUNCH_TRACKED[i - 1];
    const [h2, t2] = LAUNCH_TRACKED[i];
    if (hour <= h2) {
      const share = (Math.log(hour) - Math.log(h1)) / (Math.log(h2) - Math.log(h1));
      return Math.log(t1) + share * (Math.log(t2) - Math.log(t1));
    }
  }
  const [h1, t1] = LAUNCH_TRACKED[LAUNCH_TRACKED.length - 2];
  const [h2, t2] = LAUNCH_TRACKED[LAUNCH_TRACKED.length - 1];
  const slope = ((Math.log(t2) - Math.log(t1)) / (Math.log(h2) - Math.log(h1))) * DAMPING;
  return Math.log(t2) + slope * (Math.log(hour) - Math.log(h2));
}

/** True when no read shows any movement: the page has not refreshed inside this history. */
function isFlat(history: Read[]): boolean {
  const tracked = new Set<string>();
  for (const read of history) tracked.add(String(read.players_tracked));
  if (tracked.size > 1) return false;
  for (let index = 0; index < (history[0]?.rates.length ?? 0); index += 1) {
    const seen = new Set<string>();
    for (const read of history) seen.add(String(read.rates[index]));
    if (seen.size > 1) return false;
  }
  return true;
}

/** The last finite rate each trophy's history shows. */
function lastRates(c: Case): number[] {
  return c.trophies.map((_trophy, index) => {
    for (let i = c.history.length - 1; i >= 0; i -= 1) {
      const rate = c.history[i].rates[index];
      if (rate !== null && rate !== undefined && Number.isFinite(rate)) return rate;
    }
    return 1;
  });
}

/**
 * A weighted least-squares slope of logit(rate) on log(1 + hour), plus the robust scatter of its
 * residuals. Reads inside the trend window count in full, older ones not at all.
 */
function fitTrend(points: Point[]): { slope: number; scatter: number } | null {
  if (points.length < MIN_POINTS) return null;
  const g = points.map((p) => Math.log(1 + p.hour));
  const meanG = g.reduce((a, b) => a + b, 0) / g.length;
  const meanX = points.reduce((a, p) => a + p.x, 0) / points.length;
  let num = 0;
  let den = 0;
  for (let i = 0; i < points.length; i += 1) {
    num += (g[i] - meanG) * (points[i].x - meanX);
    den += (g[i] - meanG) ** 2;
  }
  if (!(den > 1e-9)) return null;
  const slope = num / den;
  if (!Number.isFinite(slope)) return null;
  const residual = points.map((p, i) => p.x - (meanX + slope * (g[i] - meanG)));
  const scatter = median(residual.map((r) => Math.abs(r))) * 1.4826;
  return { slope, scatter };
}

/** One trophy's five quantiles in percent, non-decreasing, centred on the projection. */
function forecastRow(projected: number, sigma: number): number[] {
  const row: number[] = [];
  for (let i = 0; i < LEVELS.length; i += 1) {
    const z = Z[i];
    const skew = SKEW * sigma * (z < 0 ? z : 0);
    row.push(round4(expitPercent(projected + z * sigma + skew)));
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
  const rates = lastRates(c);
  const hours = c.history.map((read) => hoursOf(read.at));
  const lastHour = Number.isFinite(hours[hours.length - 1]) ? hours[hours.length - 1] : 0;
  const targetHour = Number.isFinite(hoursOf(c.target)) ? hoursOf(c.target) : lastHour;
  const flat = isFlat(c.history);

  // Where the tracked count will be at the target. A history that moved is fitted and carried
  // along log(1 + hour), damped like every other growth rate; one that did not is read from the
  // page's own launch cadence, because a flat history carries no growth to measure at all.
  const trackedSeen = c.history
    .map((read, i) => ({ hour: hours[i], tracked: read.players_tracked }))
    .filter((p) => Number.isFinite(p.hour) && Number.isFinite(p.tracked) && p.tracked > 0);
  const trackedNow = trackedSeen.length > 0
    ? Math.log(trackedSeen[trackedSeen.length - 1].tracked)
    : launchLogTracked(lastHour);
  const span = Math.log(1 + targetHour) - Math.log(1 + lastHour);

  // Each trophy is projected where it physically lives: its earners, against the tracked count.
  // Earners are cumulative and never fall, so they are fitted and carried on the log scale with a
  // slope that cannot be negative; the tracked count is fitted the same way. The rate at the target
  // is then the ratio of the two, which cannot be driven past any bound the data does not support.
  const earnersNow: number[] = [];
  const earnerSeries: Point[][] = c.trophies.map((_trophy, index) => {
    const points: Point[] = [];
    c.history.forEach((read, i) => {
      const hour = hours[i];
      const rate = read.rates[index];
      const tracked = read.players_tracked;
      if (!Number.isFinite(hour) || rate === null || rate === undefined || !Number.isFinite(rate)) return;
      if (!Number.isFinite(tracked) || tracked <= 0) return;
      if (lastHour - hour > TREND_WINDOW_HOURS) return;
      // One earner is the floor: the page prints two decimals, and a rate of 0.00% still means the
      // trophy may have been earned, so it is never allowed to send the log to -infinity.
      points.push({ hour, x: Math.log(Math.max((clamp(rate) / 100) * tracked, 1)) });
    });
    earnersNow.push(points.length > 0 ? points[points.length - 1].x : Math.log(1));
    return points;
  });

  const earnerFits = earnerSeries.map((points) => fitTrend(points));
  const pooledEarnerSlope = median(
    earnerFits.filter((f) => f !== null && f.slope > 0).map((f) => Math.max(0, f!.slope)),
  );

  const trackedFit = !flat && trackedSeen.length >= MIN_POINTS
    ? fitTrend(trackedSeen.map((p) => ({ hour: p.hour, x: Math.log(p.tracked) })))
    : null;
  const trackedSlope = trackedFit === null ? 0 : Math.max(0, trackedFit.slope);
  // A history with no refresh in it carries no growth at all, so the tracked count is read from the
  // page's own launch cadence instead of fitted: that cadence is exactly what the next refresh does.
  const logTrackedTarget = flat
    ? Math.max(launchLogTracked(targetHour), trackedNow)
    : trackedNow + Math.max(0, trackedSlope * DAMPING * span);

  const rows = c.trophies.map((trophy, index) => {
    const rate = rates[index];
    const saturated = !(rate < SATURATED_HI);
    const fit = earnerFits[index];

    // Earners growth, as a share of the tracked growth. A history that moved measures it directly
    // for this trophy; one that did not falls back on how popular the trophy is, which is what the
    // page's own first refresh showed decides it.
    const measured = fit === null || !(fit.slope > 0) ? pooledEarnerSlope : fit.slope;
    const baseKeep = Number.isFinite(measured) && trackedSlope > 1e-9
      ? Math.min(1, Math.max(0, measured / trackedSlope))
      : Math.min(1, Math.max(0, KEEP_INTERCEPT + KEEP_SLOPE * logit(rate)));

    // Past the eleven hours the keep was measured over, the accounts being added are newer to the
    // game and have reached less, so a trophy keeps up with less of the growth.
    const horizonHours = Math.max(0, targetHour - lastHour);
    const decay = Math.exp(-KEEP_DECAY * Math.log(Math.max(1, horizonHours / 11)));
    const keep = baseKeep * decay;
    const gap = Math.max(0, logTrackedTarget - trackedNow);

    const logEarnersTarget = saturated
      ? logTrackedTarget
      : earnersNow[index] + keep * gap;
    let projected: number;
    if (saturated) {
      projected = logit(CLAMP_HI) - 0.02 * Math.sqrt(horizonDays);
    } else {
      // logit(100 * earners / tracked) = log(share) - log(1 - share), share = earners / tracked,
      // held a hair inside the open interval so the two ends of the scale stay finite.
      const share = Math.min(1 - 1e-6, Math.max(1e-9, Math.exp(logEarnersTarget - logTrackedTarget)));
      projected = Math.max(logit(CLAMP_LO), Math.min(logit(CLAMP_HI), Math.log(share / (1 - share))));
    }
    const scatter = fit === null ? 0 : fit.scatter;
    const sigma = Math.min(
      SIGMA_MAX,
      SIGMA_BASE
        + SIGMA_GROWTH * Math.sqrt(horizonDays)
        + (flat ? SIGMA_REFRESH : 0)
        + Math.min(0.8, scatter * Math.sqrt(horizonDays) * 0.5),
    );
    return { id: trophy.id, q: forecastRow(projected, sigma) };
  });

  writeFileSync(join(OUTPUT_DIR, `${c.id}.out`), JSON.stringify({ trophies: rows }), 'utf8');
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((name) => name.endsWith('.in')).sort();
for (const file of inputs) {
  forecastCase(JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case);
}

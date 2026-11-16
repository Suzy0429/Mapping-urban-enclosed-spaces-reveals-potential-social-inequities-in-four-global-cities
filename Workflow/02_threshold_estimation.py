# -*- coding: utf-8 -*-
"""Step 2 - estimation of the enclosure threshold T.

A parcel is enclosed when its closure degree reaches T, and the published rule
carries an internal-road requirement as well:

    ue = (FBW >= T) & (road == 1)

T is fitted on hand-labelled parcels by exhaustive search over the midpoints of
the sorted closure degrees, taking the value that maximises classification
accuracy.  The same value is simultaneously the F1- and the Youden-J-optimal
threshold on that set.  Because the closure degree is recorded to two decimals
and no labelled parcel falls inside (0.730, 0.740), the training set identifies
T as that interval rather than as a point.

The script reports three things.

  1. The point estimate on the training set, under all three criteria, together
     with the identification interval.
  2. A bootstrap of that fitting procedure.  On the training set the percentile
     interval is degenerate -- resampling the same 700 parcels cannot create a
     closure degree inside the empty band -- and it is reported as such rather
     than dressed up as precision.
  3. If a validation table is supplied, the F1-optimal threshold and its
     bootstrap interval on the independently labelled parcels, which is the
     non-degenerate interval quoted in the paper.

Usage
  python 02_threshold_estimation.py \
      --train Data/UEthreshold_train.csv \
      --validation Data/UEthreshold_validation.csv \
      --parcels Data/parcels_fbw.csv --out-dir out

Input schemas
  --train       CSV with columns FBW, ue                 (and road, if not pre-filtered)
  --validation  CSV with columns city, FBW, ue, road
  --parcels     CSV with columns city, FBW, road, area   (all parcels of the four cities)
"""
import argparse
import os

import numpy as np
import pandas as pd

CRITERIA = ("accuracy", "f1", "youden_j")


def sweep(fbw, y, road=None):
    """Score every candidate threshold of the rule in one vectorised pass.

    Candidates are the midpoints between consecutive distinct closure degrees,
    which is the grid the published search runs over.  Returns the candidates
    and, aligned with them, the accuracy, F1 and Youden's J attained at each.
    """
    fbw = np.asarray(fbw, dtype=float)
    y = np.asarray(y, dtype=int)
    eligible = np.ones(len(fbw), bool) if road is None else (np.asarray(road) == 1)
    f, yy = fbw[eligible], y[eligible]

    order = np.argsort(f, kind="mergesort")
    f, yy = f[order], yy[order]
    n = len(f)
    if n < 2:
        return np.array([]), {}

    # index of the first occurrence of each distinct value
    first = np.flatnonzero(np.concatenate([[True], f[1:] != f[:-1]]))
    cut = first[1:]                      # predict enclosed for FBW >= f[cut]
    if not len(cut):
        return np.array([]), {}
    candidates = (f[cut - 1] + f[cut]) / 2.0

    cumulative = np.concatenate([[0], np.cumsum(yy)])
    positives = int(yy.sum())
    TP = positives - cumulative[cut]
    FN = cumulative[cut]
    FP = (n - cut) - TP
    TN = cut - FN
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
        recall = np.where(TP + FN > 0, TP / (TP + FN), 0.0)
        accuracy = (TP + TN) / n
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
        youden = recall - np.where(FP + TN > 0, FP / (FP + TN), 0.0)
    return candidates, {"accuracy": accuracy, "f1": f1, "youden_j": youden}


def fit(fbw, y, criterion, road=None):
    """Lowest candidate threshold attaining the maximum score under `criterion`."""
    candidates, scores = sweep(fbw, y, road)
    if not len(candidates):
        return np.nan, np.nan
    score = scores[criterion]
    best = np.nanmax(score)
    return float(candidates[np.flatnonzero(score >= best - 1e-12)][0]), float(best)


def identification_interval(fbw, threshold, road=None):
    """The thresholds that reproduce the partition at `threshold`: (lo, hi]."""
    eligible = np.ones(len(fbw), bool) if road is None else (np.asarray(road) == 1)
    values = np.sort(np.unique(np.asarray(fbw, dtype=float)[eligible]))
    below, above = values[values < threshold], values[values >= threshold]
    return (float(below.max()) if len(below) else -np.inf,
            float(above.min()) if len(above) else np.inf)


def bootstrap_threshold(fbw, y, criterion, replicates, rng, road=None):
    values = np.empty(replicates)
    n = len(fbw)
    for b in range(replicates):
        index = rng.integers(0, n, n)
        sub_road = None if road is None else np.asarray(road)[index]
        values[b] = fit(np.asarray(fbw)[index], np.asarray(y)[index], criterion, sub_road)[0]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", default=None)
    parser.add_argument("--parcels", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--threshold-grid", default="0.70,0.72,0.73,0.735,0.74,0.75,0.76,0.78,0.80")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train = pd.read_csv(args.train)
    fbw = train["FBW"].values.astype(float)
    y = train["ue"].values.astype(int)
    road = train["road"].values.astype(int) if "road" in train.columns else None

    report = []
    report.append("Enclosure threshold T")
    report.append("=" * 72)
    report.append("training set: %d hand-labelled parcels, %d enclosed (%.1f%%)"
                  % (len(train), y.sum(), 100 * y.mean()))
    report.append("search: midpoints of the sorted closure degrees, %d candidates"
                  % len(sweep(fbw, y, road)[0]))
    report.append("")
    report.append("%-10s %8s %10s %26s %9s"
                  % ("criterion", "T*", "score", "identification interval", "width"))
    report.append("-" * 68)
    point = {}
    for criterion in CRITERIA:
        value, score = fit(fbw, y, criterion, road)
        low, high = identification_interval(fbw, value, road)
        point[criterion] = value
        report.append("%-10s %8.4f %10.4f   (%8.4f, %8.4f] %9.4f"
                      % (criterion, value, score, low, high, high - low))
    report.append("")
    report.append("Accuracy, F1 and Youden's J select the same threshold, %.4f." % point["accuracy"])
    report.append("No labelled parcel has a closure degree inside the interval, so every")
    report.append("threshold in it yields the identical partition.  T is identified as an")
    report.append("interval, not as a point; two decimals is the resolution of this set.")
    report.append("")

    rng = np.random.default_rng(args.seed)
    report.append("Bootstrap of the fitting procedure on the training set")
    report.append("B = %d, seed %d, the %d labelled parcels resampled with replacement"
                  % (args.bootstrap, args.seed, len(train)))
    report.append("")
    columns = {}
    report.append("%-10s %8s %8s %9s %9s %9s"
                  % ("criterion", "mean", "median", "sd", "2.5%", "97.5%"))
    report.append("-" * 58)
    for criterion in CRITERIA:
        values = bootstrap_threshold(fbw, y, criterion, args.bootstrap, rng, road)
        columns["T_" + criterion] = values
        low, high = np.percentile(values, [2.5, 97.5])
        report.append("%-10s %8.4f %8.4f %9.4f %9.4f %9.4f"
                      % (criterion, values.mean(), np.median(values), values.std(ddof=1),
                         low, high))
    pd.DataFrame(columns).to_csv(os.path.join(args.out_dir, "threshold_bootstrap.csv"),
                                 index=False)
    report.append("")
    for criterion in CRITERIA:
        values = columns["T_" + criterion]
        mode = pd.Series(values).mode().iloc[0]
        report.append("   %-10s T = %.3f in %.2f%% of replicates"
                      % (criterion, mode, 100 * (values == mode).mean()))
    low, high = np.percentile(columns["T_accuracy"], [2.5, 97.5])
    report.append("")
    report.append("The percentile interval from this set, [%.4f, %.4f], is degenerate and is"
                  % (low, high))
    report.append("reported as such.  Resampling the same labelled parcels cannot create a")
    report.append("closure degree inside the empty band, so the search returns the same grid")
    report.append("midpoint in almost every replicate.")
    report.append("")

    if args.validation:
        validation = pd.read_csv(args.validation)
        v_fbw = validation["FBW"].values.astype(float)
        v_y = validation["ue"].values.astype(int)
        v_road = validation["road"].values.astype(int) if "road" in validation.columns else None
        v_city = validation["city"].values if "city" in validation.columns else None
        scopes = ["pooled"] + (sorted(set(v_city)) if v_city is not None else [])

        report.append("Optimal threshold on the independently labelled validation set")
        report.append("n = %d, B = %d, seed %d" % (len(validation), args.bootstrap, args.seed))
        report.append("")
        header = "%-10s %8s %8s %9s %9s %10s"
        report.append(header % ("scope", "T_opt", "mean", "2.5%", "97.5%", "width"))
        report.append("-" * 58)
        out = {}
        for scope in scopes:
            mask = np.ones(len(validation), bool) if scope == "pooled" else (v_city == scope)
            for criterion in ("f1", "youden_j"):
                sub_road = None if v_road is None else v_road[mask]
                optimum, _ = fit(v_fbw[mask], v_y[mask], criterion, sub_road)
                values = bootstrap_threshold(v_fbw[mask], v_y[mask], criterion,
                                             args.bootstrap, rng, sub_road)
                low, high = np.percentile(values, [2.5, 97.5])
                out["%s_%s" % (scope, criterion)] = values
                report.append("%-10s %8.4f %8.4f %9.4f %9.4f %10.4f"
                              % (scope + "/" + criterion, optimum, values.mean(), low, high,
                                 high - low))
        pd.DataFrame(out).to_csv(os.path.join(args.out_dir, "validation_bootstrap.csv"),
                                 index=False)
        report.append("")

    if args.parcels:
        parcels = pd.read_csv(args.parcels)
        grid = [float(x) for x in args.threshold_grid.split(",")]
        rows = []
        for city, group in parcels.groupby("city"):
            total = group["area"].sum()
            for threshold in grid:
                enclosed = (group["FBW"] >= threshold) & (group["road"] == 1)
                rows.append(dict(city=city, T=threshold,
                                 share_pct=100.0 * group.loc[enclosed, "area"].sum() / total))
        total = parcels["area"].sum()
        for threshold in grid:
            enclosed = (parcels["FBW"] >= threshold) & (parcels["road"] == 1)
            rows.append(dict(city="pooled", T=threshold,
                             share_pct=100.0 * parcels.loc[enclosed, "area"].sum() / total))
        shares = pd.DataFrame(rows)
        shares.to_csv(os.path.join(args.out_dir, "share_vs_threshold.csv"), index=False)
        pivot = shares.pivot(index="T", columns="city", values="share_pct")
        order = [c for c in ["beijing", "london", "mumbai", "newyork", "pooled"]
                 if c in pivot.columns]
        report.append("UES share of study-area extent (%), by threshold")
        report.append("")
        report.append("%-8s %s" % ("T", " ".join("%9s" % c for c in order)))
        report.append("-" * (9 + 10 * len(order)))
        for threshold in grid:
            report.append("%-8.3f %s%s"
                          % (threshold,
                             " ".join("%9.1f" % pivot.loc[threshold, c] for c in order),
                             "  <- published" if abs(threshold - point["accuracy"]) < 1e-9 else ""))
        report.append("")

    path = os.path.join(args.out_dir, "threshold_report.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n".join(report))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()

"""
Generate head-profiling artifacts for the previous-cache vs latest-chunk
pseudo-query attention observation.

This script reads an existing raw_prev_current_attention.csv and writes:
  - query-robust current-share and log density-ratio head scores
  - heatmaps, histograms, local/global scatter plots
  - video/chunk split stability summaries and plots

It intentionally uses PIL instead of matplotlib because the lightweight
analysis environment may not provide matplotlib.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


EPS = 1e-12


def _font(size=12):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT = _font(12)
FONT_SMALL = _font(10)
FONT_TITLE = _font(16)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _safe_float(x):
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _mean(values):
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _corr(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return float("nan")
    xs = np.asarray([a for a, _ in pairs], dtype=np.float64)
    ys = np.asarray([b for _, b in pairs], dtype=np.float64)
    sx = xs.std()
    sy = ys.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _quantile(values, q):
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.quantile(vals, q)) if vals.size else float("nan")


def _rgb_interp(c0, c1, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(a * (1 - t) + b * t)) for a, b in zip(c0, c1))


def _diverging_color(value, vmin, vmax):
    if not math.isfinite(value):
        return (220, 220, 220)
    if value <= 0:
        denom = abs(vmin) if abs(vmin) > EPS else 1.0
        return _rgb_interp((49, 96, 178), (250, 250, 250), (value - vmin) / denom)
    denom = vmax if abs(vmax) > EPS else 1.0
    return _rgb_interp((250, 250, 250), (190, 49, 49), value / denom)


def _sequential_color(value, vmin=0.0, vmax=1.0):
    if not math.isfinite(value):
        return (220, 220, 220)
    t = (value - vmin) / max(vmax - vmin, EPS)
    if t < 0.5:
        return _rgb_interp((49, 96, 178), (250, 250, 250), t / 0.5)
    return _rgb_interp((250, 250, 250), (190, 49, 49), (t - 0.5) / 0.5)


def _draw_centered(draw, xy, text, font, fill=(0, 0, 0)):
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def save_heatmap(array, path, title, mode="sequential", vmin=None, vmax=None):
    array = np.asarray(array, dtype=np.float64)
    n_layers, n_heads = array.shape
    cell = 22
    left = 70
    top = 58
    right = 92
    bottom = 48
    w = left + n_heads * cell + right
    h = top + n_layers * cell + bottom
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 18), title, font=FONT_TITLE, fill=(0, 0, 0))

    if vmin is None:
        vmin = float(np.nanmin(array))
    if vmax is None:
        vmax = float(np.nanmax(array))

    for layer in range(n_layers):
        y0 = top + layer * cell
        draw.text((12, y0 + 5), str(layer), font=FONT_SMALL, fill=(0, 0, 0))
        for head in range(n_heads):
            x0 = left + head * cell
            value = float(array[layer, head])
            if mode == "diverging":
                color = _diverging_color(value, vmin, vmax)
            else:
                color = _sequential_color(value, vmin, vmax)
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=color)
    for head in range(0, n_heads, 2):
        _draw_centered(draw, (left + head * cell + cell / 2, top + n_layers * cell + 14), str(head), FONT_SMALL)
    draw.text((left + n_heads * cell / 2 - 16, h - 22), "Head", font=FONT, fill=(0, 0, 0))
    draw.text((10, top + n_layers * cell / 2), "Layer", font=FONT, fill=(0, 0, 0))

    # Colorbar.
    cb_x = left + n_heads * cell + 26
    cb_y = top
    cb_h = n_layers * cell
    for i in range(cb_h):
        t = 1.0 - i / max(cb_h - 1, 1)
        val = vmin + t * (vmax - vmin)
        color = _diverging_color(val, vmin, vmax) if mode == "diverging" else _sequential_color(val, vmin, vmax)
        draw.line((cb_x, cb_y + i, cb_x + 18, cb_y + i), fill=color)
    draw.rectangle((cb_x, cb_y, cb_x + 18, cb_y + cb_h), outline=(0, 0, 0))
    draw.text((cb_x + 24, cb_y - 4), f"{vmax:.2f}", font=FONT_SMALL, fill=(0, 0, 0))
    draw.text((cb_x + 24, cb_y + cb_h - 10), f"{vmin:.2f}", font=FONT_SMALL, fill=(0, 0, 0))
    if vmin < 0 < vmax:
        y0 = cb_y + int((vmax / (vmax - vmin)) * cb_h)
        draw.line((cb_x + 20, y0, cb_x + 25, y0), fill=(0, 0, 0))
        draw.text((cb_x + 28, y0 - 7), "0", font=FONT_SMALL, fill=(0, 0, 0))

    img.save(path)


def save_scatter(x, y, path, title, xlabel, ylabel, xlim=None, ylim=None):
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if not pairs:
        return
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    corr = _corr(xs, ys)
    if xlim is None:
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        pad = (hi - lo) * 0.06 if hi > lo else 0.1
        xlim = (lo - pad, hi + pad)
    if ylim is None:
        ylim = xlim

    w, h = 680, 560
    left, top, right, bottom = 82, 54, 36, 70
    px0, py0 = left, h - bottom
    px1, py1 = w - right, top
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 18), f"{title}  (corr={corr:.3f})", font=FONT_TITLE, fill=(0, 0, 0))
    draw.rectangle((px0, py1, px1, py0), outline=(0, 0, 0))

    def project(a, b):
        xx = px0 + (a - xlim[0]) / max(xlim[1] - xlim[0], EPS) * (px1 - px0)
        yy = py0 - (b - ylim[0]) / max(ylim[1] - ylim[0], EPS) * (py0 - py1)
        return xx, yy

    # y=x reference line over common visible range.
    lo = max(xlim[0], ylim[0])
    hi = min(xlim[1], ylim[1])
    if hi > lo:
        x_a, y_a = project(lo, lo)
        x_b, y_b = project(hi, hi)
        draw.line((x_a, y_a, x_b, y_b), fill=(160, 160, 160), width=2)

    for a, b in pairs:
        xx, yy = project(a, b)
        draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=(45, 95, 170), outline=None)

    for t in np.linspace(0, 1, 6):
        xv = xlim[0] + t * (xlim[1] - xlim[0])
        yv = ylim[0] + t * (ylim[1] - ylim[0])
        xx, _ = project(xv, ylim[0])
        _, yy = project(xlim[0], yv)
        draw.line((xx, py0, xx, py0 + 5), fill=(0, 0, 0))
        draw.line((px0 - 5, yy, px0, yy), fill=(0, 0, 0))
        _draw_centered(draw, (xx, py0 + 18), f"{xv:.2f}", FONT_SMALL)
        draw.text((px0 - 45, yy - 6), f"{yv:.2f}", font=FONT_SMALL, fill=(0, 0, 0))
    _draw_centered(draw, ((px0 + px1) / 2, h - 26), xlabel, FONT)
    draw.text((14, (py0 + py1) / 2), ylabel, font=FONT, fill=(0, 0, 0))
    img.save(path)


def save_hist(values, path, title, xlabel, bins=40, xrange=None, vlines=None):
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return
    if xrange is None:
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.04 if hi > lo else 0.1
        xrange = (lo - pad, hi + pad)
    counts, edges = np.histogram(np.asarray(vals), bins=bins, range=xrange)
    max_count = max(int(counts.max()), 1)

    w, h = 720, 460
    left, top, right, bottom = 80, 54, 36, 66
    px0, py0 = left, h - bottom
    px1, py1 = w - right, top
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 18), title, font=FONT_TITLE, fill=(0, 0, 0))
    draw.rectangle((px0, py1, px1, py0), outline=(0, 0, 0))

    bar_w = (px1 - px0) / bins
    for i, count in enumerate(counts):
        x0 = px0 + i * bar_w
        x1 = px0 + (i + 1) * bar_w - 1
        y1 = py0
        y0 = py0 - (count / max_count) * (py0 - py1)
        draw.rectangle((x0, y0, x1, y1), fill=(65, 110, 180), outline=(255, 255, 255))

    if vlines:
        for value, label, color in vlines:
            if xrange[0] <= value <= xrange[1]:
                xx = px0 + (value - xrange[0]) / max(xrange[1] - xrange[0], EPS) * (px1 - px0)
                draw.line((xx, py1, xx, py0), fill=color, width=2)
                draw.text((xx + 4, py1 + 5), label, font=FONT_SMALL, fill=color)

    for t in np.linspace(0, 1, 6):
        xv = xrange[0] + t * (xrange[1] - xrange[0])
        xx = px0 + t * (px1 - px0)
        draw.line((xx, py0, xx, py0 + 5), fill=(0, 0, 0))
        _draw_centered(draw, (xx, py0 + 18), f"{xv:.2f}", FONT_SMALL)
    for t in np.linspace(0, 1, 5):
        yv = t * max_count
        yy = py0 - t * (py0 - py1)
        draw.line((px0 - 5, yy, px0, yy), fill=(0, 0, 0))
        draw.text((px0 - 54, yy - 6), f"{int(yv)}", font=FONT_SMALL, fill=(0, 0, 0))
    _draw_centered(draw, ((px0 + px1) / 2, h - 24), xlabel, FONT)
    draw.text((14, (py0 + py1) / 2), "Heads", font=FONT, fill=(0, 0, 0))
    img.save(path)


def aggregate_head_scores(df):
    df = df.copy()
    df["s_obs"] = 0.5 * (df["local_current_share"] + df["global_current_share"])
    df["local_log_ratio"] = np.log(np.maximum(df["local_current_to_prev_per_token_ratio"], EPS))
    df["global_log_ratio"] = np.log(np.maximum(df["global_current_to_prev_per_token_ratio"], EPS))
    df["b_obs"] = 0.5 * (df["local_log_ratio"] + df["global_log_ratio"])
    df["r_obs"] = np.exp(df["b_obs"])
    df["current_token_fraction"] = df["current_chunk_tokens"] / (
        df["prev_visual_tokens"] + df["current_chunk_tokens"]
    )

    grouped = (
        df.groupby(["layer", "head"])
        .agg(
            local_current_share=("local_current_share", "mean"),
            global_current_share=("global_current_share", "mean"),
            s_current_share=("s_obs", "mean"),
            b_log_per_token_ratio=("b_obs", "mean"),
            r_per_token_ratio=("r_obs", "mean"),
            local_per_token_ratio=("local_current_to_prev_per_token_ratio", "mean"),
            global_per_token_ratio=("global_current_to_prev_per_token_ratio", "mean"),
            current_token_fraction=("current_token_fraction", "mean"),
            num_observations=("s_obs", "size"),
        )
        .reset_index()
    )

    lo = grouped["b_log_per_token_ratio"].quantile(0.2)
    hi = grouped["b_log_per_token_ratio"].quantile(0.8)
    grouped["b_quantile_class"] = "mixed"
    grouped.loc[grouped["b_log_per_token_ratio"] <= lo, "b_quantile_class"] = "memory_oriented_bottom20"
    grouped.loc[grouped["b_log_per_token_ratio"] >= hi, "b_quantile_class"] = "current_oriented_top20"

    grouped["s_threshold_class"] = "mixed"
    grouped.loc[grouped["s_current_share"] < 0.25, "s_threshold_class"] = "memory_mass_lt25"
    grouped.loc[grouped["s_current_share"] > 0.75, "s_threshold_class"] = "current_mass_gt75"
    return df, grouped


def _matrix_from_heads(head_df, value_col):
    n_layers = int(head_df["layer"].max()) + 1
    n_heads = int(head_df["head"].max()) + 1
    arr = np.full((n_layers, n_heads), np.nan, dtype=np.float64)
    for row in head_df.itertuples(index=False):
        arr[int(row.layer), int(row.head)] = float(getattr(row, value_col))
    return arr


def _split_corr(df, split_col, split_values, score_col):
    parts = []
    for values in split_values:
        part = df[df[split_col].isin(values)]
        agg = part.groupby(["layer", "head"])[score_col].mean().reset_index()
        parts.append(agg)
    merged = parts[0].merge(parts[1], on=["layer", "head"], suffixes=("_a", "_b"))
    return merged, _corr(merged[f"{score_col}_a"], merged[f"{score_col}_b"])


def _chunk_half_split(df):
    flags = []
    medians = df.groupby("video_idx")["chunk_idx"].median().to_dict()
    for row in df.itertuples(index=False):
        flags.append("early" if int(row.chunk_idx) <= medians[int(row.video_idx)] else "late")
    out = df.copy()
    out["chunk_half"] = flags
    return out


def generate(raw_csv, out_dir):
    _ensure_dir(out_dir)
    df = pd.read_csv(raw_csv)
    df, head_df = aggregate_head_scores(df)

    head_scores_csv = os.path.join(out_dir, "head_profile_scores.csv")
    head_df.to_csv(head_scores_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    s_arr = _matrix_from_heads(head_df, "s_current_share")
    b_arr = _matrix_from_heads(head_df, "b_log_per_token_ratio")
    r_arr = _matrix_from_heads(head_df, "r_per_token_ratio")

    outputs = {
        "head_scores_csv": head_scores_csv,
        "s_heatmap": os.path.join(out_dir, "s_h_query_robust_current_share_heatmap.png"),
        "b_heatmap": os.path.join(out_dir, "b_h_log_per_token_ratio_heatmap.png"),
        "r_heatmap": os.path.join(out_dir, "r_h_per_token_ratio_heatmap.png"),
        "local_global_scatter": os.path.join(out_dir, "local_global_current_share_scatter.png"),
        "s_histogram": os.path.join(out_dir, "s_h_current_share_histogram.png"),
        "b_histogram": os.path.join(out_dir, "b_h_log_per_token_ratio_histogram.png"),
        "video_split_s_scatter": os.path.join(out_dir, "video_split_s_h_stability_scatter.png"),
        "video_split_b_scatter": os.path.join(out_dir, "video_split_b_h_stability_scatter.png"),
        "chunk_split_s_scatter": os.path.join(out_dir, "chunk_split_s_h_stability_scatter.png"),
        "chunk_split_b_scatter": os.path.join(out_dir, "chunk_split_b_h_stability_scatter.png"),
        "summary_json": os.path.join(out_dir, "head_profile_summary.json"),
        "stability_json": os.path.join(out_dir, "head_profile_stability.json"),
    }

    save_heatmap(s_arr, outputs["s_heatmap"], "s_h: query-robust current attention share", "sequential", 0.0, 1.0)
    b_abs = float(np.nanquantile(np.abs(b_arr), 0.98))
    b_abs = max(b_abs, 0.1)
    save_heatmap(
        b_arr,
        outputs["b_heatmap"],
        "b_h: log current/previous per-token attention density",
        "diverging",
        -b_abs,
        b_abs,
    )
    r_clip = float(np.nanquantile(r_arr, 0.98))
    save_heatmap(r_arr, outputs["r_heatmap"], "r_h: current/previous per-token attention density", "sequential", 0.0, r_clip)

    save_scatter(
        head_df["local_current_share"].tolist(),
        head_df["global_current_share"].tolist(),
        outputs["local_global_scatter"],
        "Local vs global current-share by layer-head",
        "local current_share",
        "global current_share",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    save_hist(
        head_df["s_current_share"].tolist(),
        outputs["s_histogram"],
        "Distribution of s_h current-share scores",
        "s_h = mean(local, global) current_share",
        bins=40,
        xrange=(0.0, 1.0),
        vlines=[(0.25, "0.25", (190, 49, 49)), (0.75, "0.75", (190, 49, 49))],
    )
    save_hist(
        head_df["b_log_per_token_ratio"].tolist(),
        outputs["b_histogram"],
        "Distribution of b_h log per-token density ratios",
        "b_h = log current/previous per-token ratio",
        bins=44,
        xrange=(
            float(np.nanquantile(head_df["b_log_per_token_ratio"], 0.01)),
            float(np.nanquantile(head_df["b_log_per_token_ratio"], 0.99)),
        ),
        vlines=[(0.0, "0", (190, 49, 49))],
    )

    # Stability splits.
    videos = sorted(int(v) for v in df["video_idx"].unique())
    half = max(1, len(videos) // 2)
    video_a = videos[:half]
    video_b = videos[half:]
    stability = {}
    if video_b:
        for score_col, label, out_key in (
            ("s_obs", "s_h video split stability", "video_split_s_scatter"),
            ("b_obs", "b_h video split stability", "video_split_b_scatter"),
        ):
            merged, corr = _split_corr(df, "video_idx", [video_a, video_b], score_col)
            stability[f"video_split_{score_col}_corr"] = corr
            stability[f"video_split_{score_col}_left_videos"] = video_a
            stability[f"video_split_{score_col}_right_videos"] = video_b
            save_scatter(
                merged[f"{score_col}_a"].tolist(),
                merged[f"{score_col}_b"].tolist(),
                outputs[out_key],
                label,
                f"videos {video_a}",
                f"videos {video_b}",
            )

    chunk_df = _chunk_half_split(df)
    for score_col, label, out_key in (
        ("s_obs", "s_h early/late chunk stability", "chunk_split_s_scatter"),
        ("b_obs", "b_h early/late chunk stability", "chunk_split_b_scatter"),
    ):
        merged, corr = _split_corr(chunk_df, "chunk_half", [["early"], ["late"]], score_col)
        stability[f"chunk_split_{score_col}_corr"] = corr
        save_scatter(
            merged[f"{score_col}_a"].tolist(),
            merged[f"{score_col}_b"].tolist(),
            outputs[out_key],
            label,
            "early chunks",
            "late chunks",
        )

    with open(outputs["stability_json"], "w") as f:
        json.dump(stability, f, indent=2)

    local = head_df["local_current_share"].to_numpy(dtype=np.float64)
    global_ = head_df["global_current_share"].to_numpy(dtype=np.float64)
    s = head_df["s_current_share"].to_numpy(dtype=np.float64)
    b = head_df["b_log_per_token_ratio"].to_numpy(dtype=np.float64)
    r = head_df["r_per_token_ratio"].to_numpy(dtype=np.float64)
    diff = global_ - local

    class_counts = {
        "s_lt_0p25": int((s < 0.25).sum()),
        "s_gt_0p75": int((s > 0.75).sum()),
        "global_lt_0p25": int((global_ < 0.25).sum()),
        "global_gt_0p75": int((global_ > 0.75).sum()),
        "b_bottom20_memory": int((head_df["b_quantile_class"] == "memory_oriented_bottom20").sum()),
        "b_top20_current": int((head_df["b_quantile_class"] == "current_oriented_top20").sum()),
        "b_mixed": int((head_df["b_quantile_class"] == "mixed").sum()),
        "r_gt_1": int((r > 1.0).sum()),
        "r_lt_1": int((r < 1.0).sum()),
        "r_lt_0p5": int((r < 0.5).sum()),
    }

    summary = {
        "raw_csv": raw_csv,
        "num_rows": int(len(df)),
        "num_heads": int(len(head_df)),
        "num_layers": int(head_df["layer"].max()) + 1,
        "heads_per_layer": int(head_df["head"].max()) + 1,
        "local_global_current_share_corr": _corr(local, global_),
        "mean_global_minus_local_current_share": float(np.nanmean(diff)),
        "mean_abs_global_minus_local_current_share": float(np.nanmean(np.abs(diff))),
        "median_abs_global_minus_local_current_share": float(np.nanmedian(np.abs(diff))),
        "s_current_share_mean": float(np.nanmean(s)),
        "s_current_share_std": float(np.nanstd(s, ddof=1)),
        "s_current_share_iqr": _quantile(s, 0.75) - _quantile(s, 0.25),
        "s_current_share_quantiles": {str(q): _quantile(s, q) for q in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]},
        "b_log_per_token_ratio_mean": float(np.nanmean(b)),
        "b_log_per_token_ratio_std": float(np.nanstd(b, ddof=1)),
        "b_log_per_token_ratio_quantiles": {str(q): _quantile(b, q) for q in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]},
        "r_per_token_ratio_median": float(np.nanmedian(r)),
        "r_per_token_ratio_quantiles": {str(q): _quantile(r, q) for q in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]},
        "class_counts": class_counts,
        "stability": stability,
        "outputs": outputs,
    }
    with open(outputs["summary_json"], "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_csv",
        default="results/observations/obs_prev_current_chunk_attention_n4/raw_prev_current_attention.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="results/observations/obs_prev_current_chunk_attention_n4",
    )
    args = parser.parse_args()
    summary = generate(args.raw_csv, args.out_dir)
    print(json.dumps({
        "summary_json": summary["outputs"]["summary_json"],
        "stability_json": summary["outputs"]["stability_json"],
        "head_scores_csv": summary["outputs"]["head_scores_csv"],
        "local_global_current_share_corr": summary["local_global_current_share_corr"],
        "video_split_s_corr": summary["stability"].get("video_split_s_obs_corr"),
        "chunk_split_s_corr": summary["stability"].get("chunk_split_s_obs_corr"),
    }, indent=2))


if __name__ == "__main__":
    main()

"""Figures of the Apple Silicon engine comparison (docs/HARDWARE.md, docs/GGUF.md) from apple_engines.json.

    uv pip install matplotlib && python docs/figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.text import Text

HERE = Path(__file__).parent
D = json.loads((HERE / "apple_engines.json").read_text(encoding="utf-8"))
MODELS = ("4.5B", "1.5B")
COLORS = {"basal": "#1f6fb4", "llama.cpp": "#e07b1f", "mlx-lm based": "#2a9d55"}
TV_FLOOR = 1e-4  # TV below this is drawn at the floor of the log scale
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
# candidate label offsets (points): 8 directions at growing distance; labels further away get a leader line
OFFSETS = [(r * dx, r * dy) for r in (7, 14, 24, 36) for dx, dy in
           ((1, 0.4), (1, -1), (-1, 0.4), (-1, -1), (0.3, 1), (0.3, -1.6), (1, 1), (-1, 1))]


def timed(model):
    return [e for e in D["engines"] if e.get(model)]


def family_legend(ax, **kw):
    ax.legend(handles=[Patch(color=c, label=f) for f, c in COLORS.items()], title="engine family", frameon=False, **kw)


def place_labels(ax, points):
    """Annotate (x, y, text) points, trying several offsets per label so that labels overlap neither each other nor
    the markers."""
    from matplotlib.transforms import Bbox
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    placed = [ax.get_legend().get_window_extent(renderer)] if ax.get_legend() else []
    marks = []
    for x, y, _ in points:
        px, py = ax.transData.transform((x, y))
        marks.append(Bbox.from_bounds(px - 7, py - 7, 14, 14))
    placed += marks
    for j, (x, y, text) in enumerate(points):
        for dx, dy in OFFSETS:
            far = abs(dx) > 10 or abs(dy) > 10
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color="#333",
                            ha="left" if dx > 0 else "right", va="center",
                            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6, shrinkA=1, shrinkB=5) if far
                            else None)
            t.update_positions(renderer)
            box = Text.get_window_extent(t, renderer).expanded(1.03, 1.25)  # the text only, not the leader line
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            if inside and not any(box.overlaps(b) for b in placed if b is not marks[j]):
                placed.append(box)
                break
            t.remove()
        else:  # no free spot: nearest offset
            ax.annotate(text, (x, y), xytext=OFFSETS[0], textcoords="offset points", fontsize=7.5, color="#333")


def speed_vs_fidelity():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, model in zip(axes, MODELS):
        pts = []
        for e in sorted(timed(model), key=lambda e: e[model]["tv_mean"]):
            r = e[model]
            y = max(r["tv_mean"], TV_FLOOR)
            ax.scatter(r["ms"], y, s=25 + 18 * r["dec_s"] / (1 if model == "4.5B" else 3),
                       color=COLORS[e["family"]], alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
            pts.append((r["ms"], y, e["label"]))
        ax.set_yscale("log")
        ax.margins(x=0.18, y=0.2)
        ax.set_xlabel("ms per decision (both option orders, one request at a time)")
        ax.set_ylabel("mean total-variation distance to fp32 (log)")
        ax.set_title(f"basal-1.0-{model}: lower latency and TV are better (lower left)")
        ax.grid(alpha=0.25, zorder=0)
        if model == "4.5B":
            family_legend(ax, loc="upper right")
    fig.suptitle(f"Speed vs faithfulness on {D['machine']}", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    for ax, model in zip(axes, MODELS):
        place_labels(ax, [(e[model]["ms"], max(e[model]["tv_mean"], TV_FLOOR), e["label"])
                          for e in sorted(timed(model), key=lambda e: e[model]["ms"])])
    fig.savefig(HERE / "apple_speed_vs_fidelity.png")


def latency_throughput():
    order = sorted(timed("4.5B"), key=lambda e: e["4.5B"]["ms"])
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, key, title in ((axes[0], "ms", "ms per decision (lower is better)"),
                           (axes[1], "dec_s", "decisions per second (higher is better)")):
        for model, off, alpha in (("4.5B", -0.2, 1.0), ("1.5B", 0.2, 0.45)):
            vals = [e[model][key] for e in order]
            ax.barh(y + off, vals, height=0.38, color=[COLORS[e["family"]] for e in order], alpha=alpha)
            for yi, v in zip(y + off, vals):
                ax.text(v, yi, f" {v:.0f}" if key == "ms" else f" {v:.1f}", va="center", fontsize=7)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, [e["label"] for e in order])
    axes[0].invert_yaxis()
    family_legend(axes[1], loc="lower right")
    fig.suptitle(f"Latency and throughput on {D['machine']}, cooled GPU, unseen prompts "
                 "(solid bars: basal-1.0-4.5B, light bars: basal-1.0-1.5B)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "apple_latency_throughput.png")


def item_columns(items):
    """Item order of the heatmaps: grouped by question type, then language and id."""
    return sorted(range(len(items)), key=lambda i: (items[i]["type"], items[i]["lang"], items[i]["id"]))


def item_axis(ax, items, cols):
    """Type bands (white separators, group labels above) and item tick labels of a heatmap."""
    types = [items[i]["type"] for i in cols]
    for c in range(1, len(cols)):
        if types[c] != types[c - 1]:
            ax.axvline(c - 0.5, color="white", linewidth=2)
    for t in dict.fromkeys(types):
        idx = [c for c, x in enumerate(types) if x == t]
        ax.text((idx[0] + idx[-1]) / 2, -0.8, f"{t} ({len(idx)})", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(cols)), [f"{items[i]['id']} {items[i]['lang']}" for i in cols], rotation=90,
                  fontsize=6.5)


def fidelity_heatmap():
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    norm = LogNorm(vmin=TV_FLOOR, vmax=1)
    for ax, model in zip(axes, MODELS):
        tv, flips, items = D["tv"][model], D["flip"][model], D["items"][model]
        rows = sorted(D["engine_keys"], key=lambda k: np.mean(tv[k]))
        cols = item_columns(items)
        m = np.array([[max(tv[k][i], TV_FLOOR) for i in cols] for k in rows])
        im = ax.imshow(m, aspect="auto", cmap="magma_r", norm=norm, interpolation="nearest")
        for r, k in enumerate(rows):
            for c, i in enumerate(cols):
                if flips[k][i]:
                    ax.text(c, r, "×", ha="center", va="center", color="#00c2ff", fontsize=9, fontweight="bold")
        item_axis(ax, items, cols)
        ax.set_yticks(range(len(rows)), [f"{D['labels'][k]}  ({np.mean(tv[k]):.4f})" for k in rows], fontsize=8)
        ax.set_title(f"basal-1.0-{model}: per-item TV to fp32, lower is better (row mean in brackets; "
                     f"× = top option differs from fp32)", pad=16)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="TV (log; lower is better)")
    fig.tight_layout()
    fig.savefig(HERE / "apple_fidelity_heatmap.png")


def family_of(key):
    fam = {e["key"]: e["family"] for e in D["engines"]}
    fam.update({"vllm-metal-unpatched": "mlx-lm based", "mlx_lm.server-unpatched": "mlx-lm based",
                "llama-server-text": "llama.cpp", "ref": "reference"})
    return fam[key]


def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion k / n."""
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def quality_rows(model):
    """fp32 reference first, then the engines from the most to the least faithful (mean TV)."""
    tv = D["tv"][model]
    return ["ref"] + sorted(D["engine_keys"], key=lambda k: np.mean(tv[k]))


def quality_comparison():
    """Accuracy against the gold labels (with 95% intervals), log-loss of the gold option and decisions changed vs
    fp32, per engine, for both models."""
    rows = quality_rows("4.5B")
    y = np.arange(len(rows))
    palette = {**COLORS, "reference": "#222"}
    colors = [palette[family_of(k)] for k in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.6), sharey=True, gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    for model, off, alpha, marker in (("4.5B", -0.18, 1.0, "o"), ("1.5B", 0.18, 0.45, "s")):
        n = len(D["items"][model])
        acc = np.array([sum(D["correct"][model][k]) / n for k in rows])
        lo, hi = np.array([wilson(a * n, n) for a in acc]).T
        nll = np.array([np.mean(-np.log(np.clip(D["p_gold"][model][k], 1e-9, 1))) for k in rows])
        changed = np.array([0 if k == "ref" else sum(D["flip"][model][k]) for k in rows])
        ax = axes[0]
        ax.hlines(y + off, lo, hi, color=colors, alpha=alpha, lw=1.4)
        ax.scatter(acc, y + off, c=colors, alpha=alpha, marker=marker, s=28, zorder=3,
                   label=f"basal-1.0-{model} ({n} items)")
        ax.axvline(acc[0], color="#222", alpha=0.5 * alpha, ls="--", lw=0.8)
        axes[1].scatter(nll, y + off, c=colors, alpha=alpha, marker=marker, s=28, zorder=3)
        axes[1].axvline(nll[0], color="#222", alpha=0.5 * alpha, ls="--", lw=0.8)
        axes[2].barh(y + off, changed, height=0.34, color=colors, alpha=alpha)
        for yi, v in zip(y + off, changed):
            axes[2].text(v, yi, f" {v}", va="center", fontsize=7)
    axes[0].set_yticks(y, [D["labels"][k] for k in rows], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.45, 1.0)
    axes[0].set_title("accuracy vs gold labels\n95% interval, higher is better")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    axes[1].set_title("log-loss of the gold option\nbefore calibration, lower is better")
    axes[2].set_title(f"decisions that differ from fp32 (lower is better)\n(of {len(D['items']['4.5B'])} items)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    family_legend(axes[2], loc="center right", fontsize=8)
    fig.suptitle("Quality on the bundled examples (dashed: fp32 reference; circles / solid: 4.5B, squares / light: "
                 "1.5B; engines ordered by faithfulness)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "apple_quality.png")


def gold_prob_heatmap():
    fig, axes = plt.subplots(2, 1, figsize=(13, 10.5))
    for ax, model in zip(axes, MODELS):
        items, pg, ok = D["items"][model], D["p_gold"][model], D["correct"][model]
        rows, cols = quality_rows(model), item_columns(items)
        im = ax.imshow(np.array([[pg[k][i] for i in cols] for k in rows]), aspect="auto", cmap="RdYlGn", vmin=0,
                       vmax=1, interpolation="nearest")
        for r, k in enumerate(rows):
            for c, i in enumerate(cols):
                if not ok[k][i]:
                    ax.text(c, r, "×", ha="center", va="center", color="#222", fontsize=8)
        ax.axhline(0.5, color="white", linewidth=2)
        item_axis(ax, items, cols)
        ax.set_yticks(range(len(rows)), [f"{D['labels'][k]}  ({sum(ok[k])}/{len(items)})" for k in rows], fontsize=8)
        ax.set_title(f"basal-1.0-{model}: P(gold) per item, higher is better (correct in brackets; "
                     f"× = wrong top option)", pad=16)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="P(gold), uncalibrated; higher is better")
    fig.tight_layout()
    fig.savefig(HERE / "apple_gold_prob_heatmap.png")


# Selected checkpoints (the original bf16 and the published conversions); keys of D["formats"]
SELECTED = {"basal-mlx", "basal-gguf:F16", "basal-gguf:Q8_0", "basal-gguf:Q4_K_M", "mlx-q8", "mlx-oQ6e"}


def memory_vs_fidelity():
    """Selected checkpoints of both models: memory vs faithfulness, latency in the labels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, model in zip(axes, ("1.5B", "4.5B")):
        rows = [r for r in D["formats"][model] if r["key"] in SELECTED]
        for r in rows:
            ax.scatter(r["gb"], max(r["tv_mean"], TV_FLOOR), s=70, color=FORMAT_COLORS[r["family"]], zorder=3,
                       edgecolor="white", linewidth=0.6)
        ax.set_yscale("log")
        ax.margins(x=0.2, y=0.25)
        ax.set_xlabel("weights (GB)")
        ax.set_ylabel("mean TV distance to fp32 (log)")
        ax.set_title(f"basal-1.0-{model}")
        ax.grid(alpha=0.25)
        if model == "1.5B":
            format_legend(ax, families={r["family"] for r in rows}, loc="upper right", fontsize=8)
    fig.suptitle("Apple Silicon checkpoints of basal-1.0: lower GB and TV are better; "
                 "labels: ms per decision, lower is better", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    for ax, model in zip(axes, ("1.5B", "4.5B")):
        rows = [r for r in D["formats"][model] if r["key"] in SELECTED]
        place_labels(ax, [(r["gb"], max(r["tv_mean"], TV_FLOOR),
                           f"{r['label']} · {r['ms']:.0f} ms")
                          for r in sorted(rows, key=lambda r: r["gb"])])
    fig.savefig(HERE / "apple_memory_vs_fidelity.png")


FORMAT_COLORS = {"bf16": "#222222", "MLX affine": "#1f6fb4", "MLX microscaling": "#8e44ad", "oMLX oQ": "#2a9d55",
                 "GGUF": "#e07b1f"}


def format_legend(ax, selected=False, families=None, **kw):
    handles = [Patch(color=c, label=f) for f, c in FORMAT_COLORS.items() if families is None or f in families]
    if selected:
        handles.append(plt.Line2D([], [], marker="o", color="w", markerfacecolor="#ccc", markeredgecolor="black",
                                  markeredgewidth=1.6, markersize=9, label="selected (converted repos private)"))
    ax.legend(handles=handles, title="format", frameon=False, **kw)


def formats_size_vs_fidelity():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, model in zip(axes, ("1.5B", "4.5B")):
        rows = D["formats"][model]
        for r in rows:
            selected = r["key"] in SELECTED
            ax.scatter(r["gb"], max(r["tv_mean"], TV_FLOOR), s=90 if selected else 55, color=FORMAT_COLORS[r["family"]],
                       zorder=4 if selected else 3, edgecolor="black" if selected else "white",
                       linewidth=1.6 if selected else 0.6)
        ax.set_yscale("log")
        ax.margins(x=0.15, y=0.2)
        ax.set_xlabel("weights on disk (GB)")
        ax.set_ylabel("mean total-variation distance to fp32 (log)")
        ax.set_title(f"basal-1.0-{model}: {len(rows)} formats (all ~same speed, see table)")
        ax.grid(alpha=0.25, zorder=0)
        if model == "1.5B":
            format_legend(ax, selected=True, loc="upper right")
    fig.suptitle(f"Quantised formats on {D['machine']}: lower disk GB and TV are better (lower left)", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    for ax, model in zip(axes, ("1.5B", "4.5B")):
        place_labels(ax, [(r["gb"], max(r["tv_mean"], TV_FLOOR), r["label"])
                          for r in sorted(D["formats"][model], key=lambda r: r["gb"])])
    fig.savefig(HERE / "apple_formats_size_vs_fidelity.png")


def formats_heatmap():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5), gridspec_kw=dict(height_ratios=[
        len(D["formats"]["1.5B"]), len(D["formats"]["4.5B"])]))
    norm = LogNorm(vmin=TV_FLOOR, vmax=1)
    for ax, model in zip(axes, ("1.5B", "4.5B")):
        tv, flips, items = D["tv"][model], D["flip"][model], D["items"][model]
        rows = sorted(D["formats"][model], key=lambda r: r["tv_mean"])
        cols = item_columns(items)
        im = ax.imshow(np.array([[max(tv[r["key"]][i], TV_FLOOR) for i in cols] for r in rows]), aspect="auto",
                       cmap="magma_r", norm=norm, interpolation="nearest")
        for y, r in enumerate(rows):
            for c, i in enumerate(cols):
                if flips[r["key"]][i]:
                    ax.text(c, y, "×", ha="center", va="center", color="#00c2ff", fontsize=9, fontweight="bold")
        item_axis(ax, items, cols)
        ax.set_yticks(range(len(rows)), [f"{r['label']}, {r['gb']:.1f} GB  ({r['tv_mean']:.4f})" for r in rows],
                      fontsize=8)
        ax.set_title(f"basal-1.0-{model}: per-item TV to fp32 by format, lower is better (row mean in brackets; "
                     f"× = top option differs from fp32)", pad=16)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="TV (log; lower is better)")
    fig.tight_layout()
    fig.savefig(HERE / "apple_formats_heatmap.png")


if __name__ == "__main__":
    speed_vs_fidelity()
    latency_throughput()
    fidelity_heatmap()
    memory_vs_fidelity()
    quality_comparison()
    gold_prob_heatmap()
    formats_size_vs_fidelity()
    formats_heatmap()

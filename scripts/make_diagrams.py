#!/usr/bin/env python3
"""
Generate architecture diagrams for the RVM-EUPE paper.
Outputs three PNGs to docs/:
  arch_overview.png     — full RVM-EUPE training pipeline
  arch_gru.png          — TransformerGRU internals
  arch_ablation.png     — A1/A2/A3 multi-depth ablation variants
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "docs"
OUT.mkdir(exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
C = dict(
    eupe    = "#2563EB",   # blue  — pretrained EUPE backbone
    gru     = "#D97706",   # amber — TransformerGRU (trained)
    decoder = "#059669",   # green — MAE Decoder (trained)
    data    = "#6B7280",   # gray  — input / output data
    mask    = "#7C3AED",   # purple — masking / target
    loss    = "#DC2626",   # red   — loss
    head    = "#0891B2",   # cyan  — readout heads (eval only)
    bg      = "#F8FAFC",   # near-white background
    border  = "#1E293B",   # dark border
    text    = "#1E293B",
    dim     = "#64748B",   # muted dimension labels
    arrow   = "#475569",
    white   = "#FFFFFF",
    fpn     = "#7C3AED",
    concat  = "#B45309",
)

DPI = 180

def box(ax, x, y, w, h, label, sublabel=None, color=C["eupe"], alpha=0.92,
        fontsize=9, radius=0.018, bold=True):
    """Draw a rounded rectangle with a centred label."""
    fc = color + "22"  # 13 % opacity fill
    ec = color
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.8, edgecolor=ec, facecolor=fc,
        zorder=3,
    )
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=C["border"], zorder=4)
    if sublabel:
        ax.text(x, y - 0.025, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color=C["dim"], zorder=4, style="italic")


def arrow(ax, x0, y0, x1, y1, label=None, color=C["arrow"], lw=1.5,
          arrowstyle="-|>", labelside="top", fontsize=7.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, mutation_scale=10),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dy = 0.018 if labelside == "top" else -0.018
        ax.text(mx, my + dy, label, ha="center", va="center",
                fontsize=fontsize, color=C["dim"], zorder=5)


def curved_arrow(ax, x0, y0, x1, y1, rad=0.3, color=C["arrow"], lw=1.6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                mutation_scale=10),
                zorder=2)


def bracket(ax, x, y_bot, y_top, color=C["dim"], lw=1.0):
    """Small vertical bracket."""
    ax.plot([x, x], [y_bot, y_top], color=color, lw=lw, zorder=3)
    ax.plot([x, x + 0.008], [y_bot, y_bot], color=color, lw=lw, zorder=3)
    ax.plot([x, x + 0.008], [y_top, y_top], color=color, lw=lw, zorder=3)


# ============================================================================
# DIAGRAM 1 — Overview
# ============================================================================
def make_overview():
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=DPI)
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.96, "RVM-EUPE: Recurrent Video MAE with EUPE Backbone",
             ha="center", va="top", fontsize=13, fontweight="bold", color=C["border"])
    fig.text(0.5, 0.91, "Stage 1: encoder frozen  │  Stage 2: end-to-end with layerwise LR decay (γ = 0.75)",
             ha="center", va="top", fontsize=8.5, color=C["dim"])

    # ── Source frames ────────────────────────────────────────────────────────
    frame_xs = [0.06, 0.06, 0.06, 0.06]
    frame_ys = [0.72, 0.58, 0.44, 0.30]
    frame_labels = ["$x_{t-3}$", "$x_{t-2}$", "$x_{t-1}$", "$x_t$"]
    for fx, fy, fl in zip(frame_xs, frame_ys, frame_labels):
        box(ax, fx, fy, 0.085, 0.095, fl, color=C["data"], fontsize=9.5)

    ax.text(0.06, 0.80, "Source\nframes", ha="center", va="center",
            fontsize=8.5, color=C["dim"], style="italic")

    # Brace covering source frames
    for i, (fx, fy) in enumerate(zip(frame_xs, frame_ys)):
        if i < len(frame_xs) - 1:
            ax.annotate("", xy=(fx + 0.043, frame_ys[i+1]),
                        xytext=(fx + 0.043, fy),
                        arrowprops=dict(arrowstyle="-", color=C["dim"], lw=0.7))

    # ── EUPE encoder column ──────────────────────────────────────────────────
    enc_x = 0.23
    for fy in frame_ys:
        box(ax, enc_x, fy, 0.115, 0.095,
            "EUPE ViT-B", "$d=768$", color=C["eupe"], fontsize=8.5)
        arrow(ax, 0.103, fy, enc_x - 0.058, fy)

    ax.text(enc_x, 0.80, "Encoder\n(pretrained)", ha="center", va="center",
            fontsize=8.5, color=C["eupe"], fontweight="bold", style="italic")

    # Frozen badge
    bp = FancyBboxPatch((enc_x - 0.048, 0.845), 0.096, 0.028,
                         boxstyle="round,pad=0,rounding_size=0.008",
                         linewidth=1.2, edgecolor=C["eupe"], facecolor=C["white"], zorder=5)
    ax.add_patch(bp)
    ax.text(enc_x, 0.859, "❄ frozen  (Stage 1)", ha="center", va="center",
            fontsize=7, color=C["eupe"], zorder=6)

    # ── TransformerGRU column ────────────────────────────────────────────────
    gru_x = 0.44
    gru_ys = frame_ys
    for i, (fy, fyl) in enumerate(zip(gru_ys, ["$s_{t-3}$", "$s_{t-2}$", "$s_{t-1}$", "$s_t$"])):
        box(ax, gru_x, fy, 0.13, 0.095,
            "TransformerGRU", fyl, color=C["gru"], fontsize=8.0)
        arrow(ax, enc_x + 0.058, fy, gru_x - 0.065, fy,
              label="$e_t$  [B,196,768]", fontsize=6.5)

    ax.text(gru_x, 0.80, "Recurrent\naggreg.", ha="center", va="center",
            fontsize=8.5, color=C["gru"], fontweight="bold", style="italic")

    # State recurrence arrows (curved, on the right)
    for i in range(len(gru_ys) - 1):
        curved_arrow(ax, gru_x + 0.065, gru_ys[i], gru_x + 0.065, gru_ys[i+1],
                     rad=-0.5, color=C["gru"], lw=1.4)
    ax.text(gru_x + 0.105, 0.51, "$s_{t-1}$", fontsize=8, color=C["gru"], va="center")

    # BPTT detach note
    ax.text(gru_x + 0.065, 0.205, "detach\n(BPTT)", ha="center", fontsize=6.5,
            color=C["gru"], style="italic")

    # ── Target frame + mask ──────────────────────────────────────────────────
    tgt_x, tgt_y = 0.64, 0.72
    box(ax, tgt_x - 0.04, tgt_y, 0.08, 0.095, "$x_{t+k}$", "target frame",
        color=C["data"], fontsize=8.5)
    box(ax, tgt_x + 0.08, tgt_y, 0.085, 0.095, "95% masking",
        color=C["mask"], fontsize=8.5)
    arrow(ax, tgt_x, tgt_y, tgt_x + 0.038, tgt_y)

    ax.text(tgt_x - 0.04, 0.80, "Target\n(sampled 4–48\nframes ahead)",
            ha="center", va="center", fontsize=7.5, color=C["dim"], style="italic")

    # ── MAE Decoder ──────────────────────────────────────────────────────────
    dec_x, dec_y = 0.67, 0.44
    box(ax, dec_x, dec_y, 0.13, 0.22,
        "MAE Decoder\n8 blocks, $d$=512", color=C["decoder"], fontsize=8.5)

    # s_t → decoder (memory)
    arrow(ax, gru_x + 0.065, gru_ys[-1], dec_x - 0.065, dec_y,
          label="memory $s_t$\n[B,196,768]", fontsize=6.5, labelside="top")

    # masked tokens → decoder
    arrow(ax, tgt_x + 0.122, tgt_y, dec_x - 0.01, dec_y + 0.07,
          label="masked tokens\n[B,N·0.05,768]", fontsize=6.0, labelside="top")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_x, loss_y = 0.87, 0.44
    box(ax, loss_x, loss_y, 0.10, 0.12,
        "$\\mathcal{L} = \\|\\hat{x} - x\\|_2^2$\n(masked patches)",
        color=C["loss"], fontsize=8.0)
    arrow(ax, dec_x + 0.065, dec_y, loss_x - 0.05, loss_y,
          label="pred pixels\n[B, 0.95·N, 768]", fontsize=6.0)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        (C["eupe"],    "EUPE ViT-B (pretrained, facebook/EUPE-ViT-B)"),
        (C["gru"],     "TransformerGRU + CrossAttn blocks (trained)"),
        (C["decoder"], "MAE Decoder, 8 blocks (trained)"),
        (C["mask"],    "95% random masking on target frame"),
        (C["loss"],    "Pixel-space L2 loss, no patch normalisation"),
    ]
    for i, (col, lbl) in enumerate(legend_items):
        bx, by = 0.03, 0.175 - i * 0.033
        ax.add_patch(FancyBboxPatch((bx, by - 0.010), 0.018, 0.018,
                     boxstyle="round,pad=0,rounding_size=0.003",
                     facecolor=col + "33", edgecolor=col, linewidth=1.2, zorder=3))
        ax.text(bx + 0.024, by - 0.001, lbl, fontsize=7.0, va="center",
                color=C["border"], zorder=4)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "arch_overview.png", dpi=DPI, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close()
    print(f"  Saved: {OUT / 'arch_overview.png'}")


# ============================================================================
# DIAGRAM 2 — TransformerGRU internals
# ============================================================================
def make_gru():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.97, "TransformerGRU — Detailed Architecture",
             ha="center", va="top", fontsize=13, fontweight="bold", color=C["border"])

    # ── Input nodes ──────────────────────────────────────────────────────────
    e_x, e_y   = 0.18, 0.75   # encoder output e_t
    s_x, s_y   = 0.18, 0.32   # previous state s_{t-1}

    box(ax, e_x, e_y, 0.14, 0.08, "$e_t$", "[B, 196, 768]",
        color=C["eupe"], fontsize=10)
    box(ax, s_x, s_y, 0.14, 0.08, "$s_{t-1}$", "[B, 196, 768]",
        color=C["gru"], fontsize=10)

    # ── Gate projections ─────────────────────────────────────────────────────
    # Update gate u
    u_x, u_y = 0.42, 0.75
    box(ax, u_x, u_y, 0.16, 0.12,
        "Update gate\n$u_t = \\sigma(W_e^u e_t + W_s^u s_{t-1})$",
        color=C["gru"], fontsize=8.0)
    arrow(ax, e_x + 0.07, e_y, u_x - 0.08, u_y, label="Linear(768,768)", fontsize=7)
    arrow(ax, s_x + 0.07, s_y, u_x - 0.08, u_y, label="Linear(768,768)", fontsize=7)

    # Reset gate r
    r_x, r_y = 0.42, 0.50
    box(ax, r_x, r_y, 0.16, 0.12,
        "Reset gate\n$r_t = \\sigma(W_e^r e_t + W_s^r s_{t-1})$",
        color=C["gru"], fontsize=8.0)
    arrow(ax, e_x + 0.07, e_y,   r_x - 0.08, r_y, fontsize=7)
    arrow(ax, s_x + 0.07, s_y,   r_x - 0.08, r_y, fontsize=7)

    # Gated state
    gs_x, gs_y = 0.42, 0.28
    box(ax, gs_x, gs_y, 0.16, 0.08,
        "$r_t \\odot s_{t-1}$", "gated prev. state",
        color=C["gru"], fontsize=8.5)
    arrow(ax, r_x + 0.08, r_y, gs_x - 0.01, gs_y + 0.035, fontsize=7)
    arrow(ax, s_x + 0.07, s_y, gs_x - 0.08, gs_y, fontsize=7)

    # ── TransformerBlock stack ────────────────────────────────────────────────
    tb_x, tb_y = 0.68, 0.56
    # Background panel
    panel = FancyBboxPatch((tb_x - 0.12, tb_y - 0.28), 0.24, 0.56,
                            boxstyle="round,pad=0,rounding_size=0.015",
                            linewidth=1.5, edgecolor=C["gru"] + "88",
                            facecolor=C["gru"] + "0C", zorder=2)
    ax.add_patch(panel)
    ax.text(tb_x, tb_y + 0.30, "Transformer Block  ×N", ha="center",
            fontsize=8.5, color=C["gru"], fontweight="bold")

    sub_y_offsets = [0.18, 0.03, -0.14]
    sub_labels = ["Cross-Attention\n(q=$e_t$, kv=$r_t \\odot s_{t-1}$)",
                  "Feed-Forward\nNetwork",
                  "Self-Attention"]
    sub_colors = [C["decoder"], C["gru"], C["gru"]]
    for dy, lbl, col in zip(sub_y_offsets, sub_labels, sub_colors):
        box(ax, tb_x, tb_y + dy, 0.20, 0.11, lbl, color=col, fontsize=8.0)

    # Residual connections (right side)
    for i in range(len(sub_y_offsets) - 1):
        y0 = tb_y + sub_y_offsets[i] - 0.055
        y1 = tb_y + sub_y_offsets[i+1] + 0.055
        ax.annotate("", xy=(tb_x, y1), xytext=(tb_x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C["gru"],
                                   lw=1.2, mutation_scale=8), zorder=3)

    # Pre-norm labels
    for dy in sub_y_offsets:
        ax.text(tb_x - 0.13, tb_y + dy, "LN", ha="center", va="center",
                fontsize=7, color=C["dim"],
                bbox=dict(boxstyle="round,pad=0.15", facecolor=C["white"],
                          edgecolor=C["dim"], linewidth=0.8))

    # ĥ_t label
    ax.text(tb_x + 0.135, tb_y - 0.15, "$\\hat{h}_t$", fontsize=10,
            color=C["gru"], va="center")

    # Arrows into transformer block
    arrow(ax, e_x + 0.07, e_y, tb_x - 0.12, tb_y + 0.18, label="query $e_t$", fontsize=7)
    arrow(ax, gs_x + 0.08, gs_y, tb_x - 0.12, tb_y + 0.18,
          label="key/value", fontsize=7)

    # ── GRU update equation ──────────────────────────────────────────────────
    up_x, up_y = 0.68, 0.18
    box(ax, up_x, up_y, 0.30, 0.12,
        "$s_t = (1 - u_t) \\odot s_{t-1}\\; +\\; u_t \\odot \\hat{h}_t$",
        color=C["gru"], fontsize=8.5)
    arrow(ax, u_x + 0.08, u_y, up_x - 0.10, up_y + 0.04,
          label="$u_t$", fontsize=8)
    arrow(ax, tb_x, tb_y - 0.28, up_x, up_y + 0.06,
          label="$\\hat{h}_t$", fontsize=8)
    arrow(ax, s_x + 0.07, s_y, up_x - 0.15, up_y, fontsize=7)

    # s_t output
    st_x, st_y = 0.68, 0.06
    box(ax, st_x, st_y, 0.14, 0.07, "$s_t$", "[B, 196, 768]",
        color=C["gru"], fontsize=10)
    arrow(ax, up_x, up_y - 0.06, st_x, st_y + 0.035, label="new state", fontsize=7)

    # Detach annotation
    ax.text(st_x + 0.11, st_y, "detach()\n(truncated\nBPTT)", ha="left",
            va="center", fontsize=7, color=C["dim"], style="italic")

    # FlashAttn note
    ax.text(tb_x, tb_y - 0.34,
            "All attention via F.scaled_dot_product_attention  (Flash-Attention 2 backend)",
            ha="center", fontsize=7.2, color=C["dim"], style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "arch_gru.png", dpi=DPI, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close()
    print(f"  Saved: {OUT / 'arch_gru.png'}")


# ============================================================================
# DIAGRAM 3 — Multi-depth ablation A1 / A2 / A3
# ============================================================================
def make_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), dpi=DPI)
    fig.patch.set_facecolor(C["bg"])

    fig.text(0.5, 0.97, "Multi-Depth Feature Ablation: A1 / A2 / A3",
             ha="center", va="top", fontsize=13, fontweight="bold", color=C["border"])
    fig.text(0.5, 0.925,
             "EUPE ViT has uniform stride-16; all variants share patch resolution H/16 × W/16 — "
             "ablation explores multi-semantic-depth fusion, not spatial pyramid",
             ha="center", va="top", fontsize=8.5, color=C["dim"])

    configs = [
        dict(
            title="A1 — Last Layer Only",
            subtitle="(Primary model)",
            layers=[11],
            adapter="None\n(direct)",
            adapter_color=C["eupe"],
            extra="forward_features()\n→ x_norm_patchtokens\n[B, 196, 768]",
            col=C["eupe"],
        ),
        dict(
            title="A2 — FPN Adapter",
            subtitle="(Multi-depth ablation)",
            layers=[2, 5, 8, 11],
            adapter="FPNAdapter\n1×1 conv/layer\n→ sum → flatten",
            adapter_color=C["fpn"],
            extra="get_intermediate_layers\n(n=[2,5,8,11], reshape=True)\n4×[B,768,H/16,W/16]",
            col=C["fpn"],
        ),
        dict(
            title="A3 — Concat Adapter",
            subtitle="(Multi-depth ablation)",
            layers=[2, 5, 8, 11],
            adapter="ConcatAdapter\nchannel concat\n→ Linear(4·768, 768)",
            adapter_color=C["concat"],
            extra="get_intermediate_layers\n(n=[2,5,8,11], reshape=True)\n4×[B,768,H/16,W/16]",
            col=C["concat"],
        ),
    ]

    for ax, cfg in zip(axes, configs):
        ax.set_facecolor(C["bg"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        col = cfg["col"]

        # Title
        ax.text(0.5, 0.95, cfg["title"], ha="center", va="top",
                fontsize=10, fontweight="bold", color=col)
        ax.text(0.5, 0.875, cfg["subtitle"], ha="center", va="top",
                fontsize=8, color=C["dim"], style="italic")

        # Input image
        box(ax, 0.5, 0.79, 0.40, 0.07, "Input frame  [B, 3, H, W]",
            color=C["data"], fontsize=8, bold=False)

        # ViT blocks (draw as a stack of rects)
        vit_top, vit_bot = 0.70, 0.30
        vit_h = (vit_top - vit_bot) / 12

        for blk in range(12):
            blk_y = vit_bot + (11 - blk) * vit_h + vit_h / 2
            highlight = (blk in cfg["layers"])
            fc = col + "44" if highlight else C["eupe"] + "11"
            ec = col if highlight else C["eupe"] + "55"
            lw = 1.8 if highlight else 0.6
            p = FancyBboxPatch((0.22, blk_y - vit_h * 0.45), 0.56, vit_h * 0.88,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
            ax.add_patch(p)
            label = f"Block {blk}" + (" ◀" if highlight else "")
            ax.text(0.50, blk_y, label, ha="center", va="center",
                    fontsize=6.2, color=col if highlight else C["dim"],
                    fontweight="bold" if highlight else "normal", zorder=4)

        ax.text(0.12, (vit_top + vit_bot) / 2, "EUPE\nViT-B\n(12 blocks)",
                ha="center", va="center", fontsize=7.5, color=C["eupe"],
                fontweight="bold")

        arrow(ax, 0.5, 0.755, 0.5, vit_top)

        # Feature extraction note
        ax.text(0.5, 0.265, cfg["extra"], ha="center", va="top",
                fontsize=6.8, color=col, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col + "11",
                          edgecolor=col + "55", linewidth=0.8))

        arrow(ax, 0.5, vit_bot, 0.5, 0.235)

        # Adapter
        box(ax, 0.5, 0.16, 0.50, 0.075, cfg["adapter"],
            color=cfg["adapter_color"], fontsize=8.0)

        arrow(ax, 0.5, 0.192, 0.5, 0.122)

        # Output
        box(ax, 0.5, 0.085, 0.50, 0.065, "Tokens  [B, 196, 768]  → TransformerGRU",
            color=C["gru"], fontsize=7.5, bold=False)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "arch_ablation.png", dpi=DPI, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close()
    print(f"  Saved: {OUT / 'arch_ablation.png'}")


# ============================================================================
# DIAGRAM 4 — Frozen-encoder evaluation pipeline
# ============================================================================
def make_eval():
    fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.96, "Frozen-Encoder Evaluation Pipeline",
             ha="center", va="top", fontsize=13, fontweight="bold", color=C["border"])
    fig.text(0.5, 0.88,
             "Full model frozen after pretraining.  Per-task readout heads trained on top of hidden state $s_t$.",
             ha="center", va="top", fontsize=8.5, color=C["dim"])

    # ── Frozen model block ────────────────────────────────────────────────────
    frozen_x, frozen_y, frozen_w, frozen_h = 0.22, 0.52, 0.32, 0.42

    panel = FancyBboxPatch((frozen_x - frozen_w/2, frozen_y - frozen_h/2),
                            frozen_w, frozen_h,
                            boxstyle="round,pad=0,rounding_size=0.015",
                            linewidth=2.0, edgecolor=C["dim"],
                            linestyle="--",
                            facecolor=C["dim"] + "08", zorder=2)
    ax.add_patch(panel)
    ax.text(frozen_x, frozen_y + frozen_h/2 + 0.04, "❄  Frozen RVM-EUPE",
            ha="center", fontsize=9, color=C["dim"], fontweight="bold")

    box(ax, frozen_x, 0.66, 0.24, 0.09, "EUPE ViT-B", "$e_t$ [B, 196, 768]",
        color=C["eupe"], fontsize=8.5)
    box(ax, frozen_x, 0.51, 0.24, 0.09, "TransformerGRU", "$s_t$ [B, 196, 768]",
        color=C["gru"], fontsize=8.5)
    box(ax, frozen_x, 0.36, 0.24, 0.09, "MAE Decoder", "(not used in eval)",
        color=C["decoder"], fontsize=8.5)
    ax.text(frozen_x + 0.185, 0.36, "✗", ha="center", va="center",
            fontsize=14, color=C["dim"])

    arrow(ax, frozen_x, 0.615, frozen_x, 0.555)
    arrow(ax, frozen_x, 0.465, frozen_x, 0.405)

    # Input
    box(ax, frozen_x, 0.84, 0.24, 0.075, "Video clip\n(4 source frames)",
        color=C["data"], fontsize=8.5)
    arrow(ax, frozen_x, 0.802, frozen_x, 0.705)

    # s_t arrow out
    arrow(ax, frozen_x + 0.16, 0.51, 0.41, 0.51,
          label="$s_t$  [B, 196, 768]", fontsize=7.5)

    # ── Readout heads ─────────────────────────────────────────────────────────
    tasks = [
        ("DAVIS 2017",        "k-NN label\npropagation\n(k=7, τ=0.7)",    "J&F mean",    0.88),
        ("Kinetics-400/700",  "AttentiveReadout\n1 learned query",         "Top-1 Acc",   0.73),
        ("SSv2",              "AttentiveReadout\n1 learned query",         "Top-1 Acc",   0.58),
        ("ScanNet Depth",     "128 spatial\nFourier queries",              "AbsRel ↓",    0.43),
        ("Waymo Tracking",    "Box-coord\nquery embeds",                   "Track metric",0.28),
        ("JHMDB Keypoints",   "k-NN, 32\njoint queries",                  "PCK@0.1",     0.13),
    ]

    head_x = 0.62
    for tname, thead, tmetric, ty in tasks:
        # Task box
        box(ax, head_x, ty, 0.20, 0.095, tname, thead, color=C["head"], fontsize=7.5)
        # Arrow from s_t output line
        arrow(ax, 0.41, 0.51, head_x - 0.10, ty, color=C["arrow"], lw=1.2)
        # Metric badge
        ax.text(head_x + 0.13, ty, tmetric, ha="left", va="center",
                fontsize=7, color=C["head"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor=C["head"] + "18",
                          edgecolor=C["head"] + "66", linewidth=0.8))

    ax.text(head_x, 0.95, "Readout Heads  (trained, encoder frozen)",
            ha="center", fontsize=9, color=C["head"], fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(OUT / "arch_eval.png", dpi=DPI, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close()
    print(f"  Saved: {OUT / 'arch_eval.png'}")


if __name__ == "__main__":
    print("Generating architecture diagrams...")
    make_overview()
    make_gru()
    make_ablation()
    make_eval()
    print("Done.")

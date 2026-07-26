"""
plot_results.py — Vẽ 4 biểu đồ kết quả huấn luyện DQN từ file CSV logs/train_log.csv.

Biểu đồ được vẽ:
  (a) Total Reward theo Episode (raw + moving average 100 ep)
  (e) Rolling Success Rate 200 episode theo thời gian
  (g) Phân bố kết quả cuối (Success / Collision / Timeout) - bar chart
  (x) Phân bố kết quả theo từng Map Stage (stacked bar)

Kết quả ảnh lưu vào: results/charts/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ── Cấu hình ──────────────────────────────────────────────────
LOG_CSV   = "results_ppo/train_log.csv"
CHART_DIR = "results_ppo/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# Bảng màu thống nhất
COLOR_SUCCESS   = "#2ecc71"    # xanh lá
COLOR_COLLISION = "#e74c3c"    # đỏ
COLOR_TIMEOUT   = "#f39c12"    # cam vàng
COLOR_RAW       = "#bdc3c7"    # xám nhạt
COLOR_MA        = "#2980b9"    # xanh đậm
COLOR_SR        = "#8e44ad"    # tím

STAGE_COLORS = ["#3498db", "#e67e22", "#c0392b"]  # easy / medium / hard
STAGE_LABELS = ["Stage 1: map_easy", "Stage 2: map_medium", "Stage 3: map_hard"]

FONT_TITLE  = 15
FONT_LABEL  = 12
FONT_TICK   = 10
DPI         = 150


def load_data() -> pd.DataFrame:
    """Nạp CSV và kiểm tra dữ liệu cơ bản."""
    csv_path = LOG_CSV
    if not os.path.exists(csv_path) and os.path.exists("results_ppo/logs/train_log.csv"):
        csv_path = "results_ppo/logs/train_log.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Không tìm thấy '{LOG_CSV}' hoặc 'results_ppo/logs/train_log.csv'. Hãy chạy train_ppo.py trước."
        )
    df = pd.read_csv(csv_path)
    required = {"episode", "timestep", "total_reward", "episode_length",
                "outcome", "current_map_stage", "rolling_success_rate_200"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu các cột: {missing}")
    df = df.dropna(subset=["episode", "total_reward", "outcome"])
    print(f"[Plot] Nạp {len(df)} episodes từ {LOG_CSV}")
    return df


def moving_average(data, window=100):
    """Tính moving average với cửa sổ trượt, padding đầu bằng NaN."""
    ma = np.full(len(data), np.nan)
    for i in range(window - 1, len(data)):
        ma[i] = np.mean(data[i - window + 1: i + 1])
    return ma


# ── Biểu đồ (a): Reward theo Episode ─────────────────────────
def plot_reward(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))

    episodes = df["episode"].values
    rewards  = df["total_reward"].values
    ma_100   = moving_average(rewards, window=100)

    # Vùng nền theo stage
    stage_changes = df.groupby("current_map_stage")["episode"].agg(["min", "max"])
    for stage_idx, row in stage_changes.iterrows():
        if stage_idx < len(STAGE_COLORS):
            ax.axvspan(row["min"], row["max"],
                       alpha=0.07, color=STAGE_COLORS[stage_idx],
                       label=f"_{STAGE_LABELS[stage_idx]}")

    # Raw reward
    ax.plot(episodes, rewards, color=COLOR_RAW, linewidth=0.6,
            alpha=0.6, label="Raw Reward (mỗi episode)")
    # Moving average
    ax.plot(episodes, ma_100, color=COLOR_MA, linewidth=2.0,
            label="Moving Average (100 ep)")

    # Đường tham chiếu
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    # Stage transition markers
    for stage_idx, row in stage_changes.iterrows():
        if stage_idx > 0:
            ax.axvline(row["min"], color=STAGE_COLORS[stage_idx],
                       linestyle=":", linewidth=1.5, alpha=0.8)
            ax.text(row["min"] + 5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] != 1.0 else 20,
                    STAGE_LABELS[stage_idx].split(":")[1].strip(),
                    fontsize=8, color=STAGE_COLORS[stage_idx], va="top")

    # Legend patches cho stage
    stage_patches = [Patch(facecolor=STAGE_COLORS[i], alpha=0.3, label=STAGE_LABELS[i])
                     for i in range(min(3, len(stage_changes)))]

    handles, labels = ax.get_legend_handles_labels()
    labels_clean = [l for l in labels if not l.startswith("_")]
    handles_clean = [h for h, l in zip(handles, labels) if not l.startswith("_")]
    ax.legend(handles_clean + stage_patches, labels_clean + [s.get_label() for s in stage_patches],
              fontsize=FONT_TICK, loc="upper left")

    ax.set_title("(a) Tổng Reward theo Episode", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=FONT_LABEL)
    ax.set_ylabel("Total Reward", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(CHART_DIR, "a_reward_per_episode.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[Plot] Đã lưu: {out}")


# ── Biểu đồ (e): Rolling Success Rate ────────────────────────
def plot_rolling_success_rate(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))

    episodes = df["episode"].values
    sr_200   = df["rolling_success_rate_200"].values * 100  # chuyển sang %

    # Vùng nền theo stage
    stage_changes = df.groupby("current_map_stage")["episode"].agg(["min", "max"])
    for stage_idx, row in stage_changes.iterrows():
        if stage_idx < len(STAGE_COLORS):
            ax.axvspan(row["min"], row["max"],
                       alpha=0.07, color=STAGE_COLORS[stage_idx])

    # Đường rolling SR
    ax.plot(episodes, sr_200, color=COLOR_SR, linewidth=1.8,
            label="Rolling Success Rate (cửa sổ 200 ep)")

    # Ngưỡng curriculum (70%) và early stop (85%)
    ax.axhline(70, color=COLOR_COLLISION, linestyle="--", linewidth=1.2,
               alpha=0.8, label="Ngưỡng chuyển stage: 70%")
    ax.axhline(85, color=COLOR_SUCCESS, linestyle="--", linewidth=1.2,
               alpha=0.8, label="Ngưỡng Early Stop: 85%")

    # Stage transition markers
    for stage_idx, row in stage_changes.iterrows():
        if stage_idx > 0:
            ax.axvline(row["min"], color=STAGE_COLORS[stage_idx],
                       linestyle=":", linewidth=1.5, alpha=0.8)

    # Legend stage patches
    stage_patches = [Patch(facecolor=STAGE_COLORS[i], alpha=0.3, label=STAGE_LABELS[i])
                     for i in range(min(3, len(stage_changes)))]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + stage_patches, labels + [s.get_label() for s in stage_patches],
              fontsize=FONT_TICK, loc="lower right")

    ax.set_title("(e) Rolling Success Rate theo Episode (cửa sổ 200 episode)",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=FONT_LABEL)
    ax.set_ylabel("Success Rate (%)", fontsize=FONT_LABEL)
    ax.set_ylim(-2, 102)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(CHART_DIR, "e_rolling_success_rate.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[Plot] Đã lưu: {out}")


# ── Biểu đồ (g): Phân bố Outcome toàn bộ training ───────────
def plot_outcome_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Trái: phân bố toàn bộ ──
    ax = axes[0]
    counts = df["outcome"].value_counts()
    labels = ["success", "collision", "timeout"]
    values = [counts.get(l, 0) for l in labels]
    colors = [COLOR_SUCCESS, COLOR_COLLISION, COLOR_TIMEOUT]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=FONT_TICK, fontweight="bold")

    ax.set_title("(g) Phân Bố Kết Quả — Toàn Bộ Training",
                 fontsize=FONT_TITLE - 1, fontweight="bold")
    ax.set_ylabel("Số Episode", fontsize=FONT_LABEL)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["Success\n(Đến đích)", "Collision\n(Va chạm)", "Timeout\n(Hết bước)"],
                       fontsize=FONT_TICK)
    ax.grid(axis="y", alpha=0.3)

    # ── Phải: phân bố theo stage ──
    ax2 = axes[1]
    stages = sorted(df["current_map_stage"].unique())
    x = np.arange(len(stages))
    width = 0.25

    for i, (outcome, color) in enumerate(zip(labels, colors)):
        vals = []
        for s in stages:
            stage_df = df[df["current_map_stage"] == s]
            c = (stage_df["outcome"] == outcome).sum()
            vals.append(c / len(stage_df) * 100 if len(stage_df) > 0 else 0)
        bars2 = ax2.bar(x + i * width, vals, width,
                        label=outcome.capitalize(), color=color,
                        edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars2, vals):
            if val > 2:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.5,
                         f"{val:.0f}%", ha="center", va="bottom",
                         fontsize=8)

    stage_x_labels = [STAGE_LABELS[s] if s < len(STAGE_LABELS) else f"Stage {s}"
                      for s in stages]
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(stage_x_labels, fontsize=FONT_TICK - 1, rotation=10)
    ax2.set_ylabel("Tỷ lệ (%)", fontsize=FONT_LABEL)
    ax2.set_title("(g+) Phân Bố Kết Quả Theo Giai Đoạn Curriculum",
                  fontsize=FONT_TITLE - 1, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.legend(fontsize=FONT_TICK, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    out = os.path.join(CHART_DIR, "g_outcome_distribution.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[Plot] Đã lưu: {out}")


# ── Biểu đồ (x): Episode Length theo stage ────────────────────
def plot_episode_length(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))

    episodes = df["episode"].values
    ep_len   = df["episode_length"].values
    ma_50    = moving_average(ep_len, window=50)

    stage_changes = df.groupby("current_map_stage")["episode"].agg(["min", "max"])
    for stage_idx, row in stage_changes.iterrows():
        if stage_idx < len(STAGE_COLORS):
            ax.axvspan(row["min"], row["max"],
                       alpha=0.07, color=STAGE_COLORS[stage_idx])

    ax.plot(episodes, ep_len, color=COLOR_RAW, linewidth=0.5, alpha=0.5,
            label="Episode Length (raw)")
    ax.plot(episodes, ma_50, color="#e67e22", linewidth=2.0,
            label="Moving Average (50 ep)")

    stage_patches = [Patch(facecolor=STAGE_COLORS[i], alpha=0.3, label=STAGE_LABELS[i])
                     for i in range(min(3, len(stage_changes)))]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + stage_patches, labels + [s.get_label() for s in stage_patches],
              fontsize=FONT_TICK)

    ax.set_title("(x) Episode Length theo Episode", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=FONT_LABEL)
    ax.set_ylabel("Số bước (Steps)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(CHART_DIR, "x_episode_length.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[Plot] Đã lưu: {out}")


def main():
    print("=" * 60)
    print("  UAV PPO — Vẽ Biểu Đồ Kết Quả Huấn Luyện")
    print("=" * 60)

    df = load_data()

    print(f"\n[Stats] Tổng episode   : {len(df):,}")
    print(f"[Stats] Tổng timesteps  : {df['timestep'].max():,}")
    outcomes = df["outcome"].value_counts()
    for k, v in outcomes.items():
        print(f"[Stats] {k:<12}: {v:>6,} ({v/len(df):.1%})")

    plot_reward(df)
    plot_rolling_success_rate(df)
    plot_outcome_distribution(df)
    plot_episode_length(df)

    print(f"\n[Done] Tất cả biểu đồ đã lưu vào: {CHART_DIR}/")


if __name__ == "__main__":
    main()

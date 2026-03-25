import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

st.set_page_config(layout="wide", page_title="Pass Map + Statistics")

st.title("Pass Map + Statistics (All Matches)")

# ==========================
# CONFIG
# ==========================
GOAL_X = 120
GOAL_Y = 40
FINAL_THIRD_LINE_X = 80  # Final third line: x=80 (entry: start < 80 and end >= 80)
PROGRESSIVE_THRESHOLD = 0.75  # Opta progressive rule (your notebook)

# ==========================
# INPUT (coords + passes_errados) - already compiled
# coords = [start1, end1, start2, end2, ...]
# passes_errados = 1-indexed pass numbers that are "errado"
# ==========================
coords = [
    (29.25, 20.38),(23.76, 12.9),(14.78, 31.52),(32.57, 41.99),
    (70.64, 55.13),(75.29, 58.29),(43.04, 50.14),(30.58, 50.31),
    (50.86, 49.14),(45.04, 58.62),(37.56, 58.62),(55.18, 50.47),
    (43.54, 39.17),(57.67, 32.85),(55.35, 30.69),(37.39, 25.37),
    (32.9, 52.97),(41.05, 41.33),(65.82, 40.83),(48.2, 42.66),
    (27.58, 52.47),(13.62, 51.47),(33.57, 39.83),(32.57, 25.7),
    (50.19, 53.13),(44.87, 61.78),(53.35, 47.81),(48.2, 56.79),
    (54.85, 44.99),(55.18, 37.01),(54.18, 73.25),(66.15, 78.57),
    (22.1, 28.36),(36.56, 23.88),(59.83, 61.61),(70.31, 53.96),
    (31.08, 59.28),(38.22, 70.75),(51.02, 54.46),(60.66, 50.64),
    (55.35, 48.15),(66.48, 74.74),(59.17, 64.27),(50.69, 35.51),
    (58.5, 47.48),(54.35, 31.69),(78.29, 59.28),(87.59, 75.57),
    (52.52, 57.12),(45.7, 40.33),(28.91, 61.11),(44.37, 56.62),
    (46.04, 55.46),(45.21, 33.68),(69.14, 57.12),(64.16, 40.66),
    (29.75, 57.29),(34.57, 75.91),(29.91, 32.02),(39.39, 46.98),
    (90.25, 58.12),(98.57, 63.44),(94.24, 64.44),(95.41, 75.57),
    (26.92, 26.37),(17.11, 26.7),(58.67, 32.85),(71.3, 39.5),
    (76.12, 45.49),(91.25, 47.98),(102.89, 47.15),(98.4, 41.66),
    (73.8, 71.09),(76.29, 76.07),(82.94, 48.31),(95.24, 51.14),
    (59.67, 48.48),(70.31, 71.92),(75.96, 37.34),(88.76, 31.85),
    (44.37, 28.03),(37.23, 53.46),(44.71, 18.72),(39.72, 10.58),
    (37.89, 44.99),(21.77, 40.83),(34.57, 35.84),(46.04, 36.51),
    (28.58, 53.13),(36.56, 61.11),(34.4, 23.88),(25.42, 35.51),
    (64.99, 35.68),(76.29, 11.57),(38.22, 39.83),(29.08, 23.21),
    (75.46, 31.02),(65.15, 45.98),(28.25, 35.84),(32.9, 20.05),
    (29.75, 24.71),(17.44, 18.56),(44.04, 49.64),(53.35, 58.95),
    (56.68, 37.34),(43.88, 34.02),(40.22, 41.5),(23.59, 50.14),
    (77.95, 31.36),(92.25, 48.31),(85.27, 61.28),(96.07, 66.1),
    (34.4, 38.67),(26.59, 24.87),(44.21, 13.9),(34.23, 16.89),
    (45.87, 33.68),(37.06, 19.05),(55.68, 56.12),(71.3, 73.58),
    (30.74, 34.35),(16.61, 27.03),(64.65, 66.76),(69.97, 73.58),
    (33.07, 49.48),(24.59, 55.46),(48.2, 54.13),(60.33, 54.96),
    (38.89, 29.53),(30.74, 51.97),(55.18, 26.04),(67.98, 45.98),
    (57.84, 34.68),(71.97, 37.34),(36.89, 39.0),(58.5, 66.1),
    (67.15, 36.18),(75.13, 24.21),(25.76, 31.85),(11.79, 18.39),
    (68.31, 32.19),(54.68, 38.17),(57.17, 45.65),(69.48, 46.82),
    (81.11, 12.41),(83.94, 15.06),(64.99, 41.83),(69.64, 79.9),
    (42.05, 29.19),(46.87, 5.26)
]

passes_errados = [3, 15, 38, 52, 62, 67, 68, 72, 73, 74]  # 1-indexed

# ==========================
# Build DataFrame
# ==========================
passes = []
for i in range(0, len(coords), 2):
    start = coords[i]
    end = coords[i + 1]
    numero = i // 2 + 1
    passes.append(
        {
            "numero": numero,
            "x_start": float(start[0]),
            "y_start": float(start[1]),
            "x_end": float(end[0]),
            "y_end": float(end[1]),
        }
    )

df = pd.DataFrame(passes)
df["errado"] = df["numero"].isin(passes_errados)
df["certo"] = ~df["errado"]

# Opta-like progressive rule
dist_inicio = np.sqrt((GOAL_X - df["x_start"]) ** 2 + (GOAL_Y - df["y_start"]) ** 2)
dist_fim = np.sqrt((GOAL_X - df["x_end"]) ** 2 + (GOAL_Y - df["y_end"]) ** 2)
df["progressivo"] = dist_fim <= dist_inicio * PROGRESSIVE_THRESHOLD

# Final third (entry): starts outside and ends inside
df["into_final_third"] = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)

# Directions
df["pra_frente"] = df["x_end"] > df["x_start"]
df["pra_tras"] = df["x_end"] < df["x_start"]
df["pra_direita"] = df["y_end"] > df["y_start"]
df["pra_esquerda"] = df["y_end"] < df["y_start"]

# ==========================
# Statistics
# ==========================
total_passes = len(df)
passes_certs = int(df["certo"].sum())
passes_errs = int(df["errado"].sum())

perc_accuracy = (passes_certs / total_passes * 100.0) if total_passes else 0.0

passes_into_final_third = int(df["into_final_third"].sum())
passes_into_final_third_certs = int((df["into_final_third"] & ~df["errado"]).sum())
passes_into_final_third_errs = int((df["into_final_third"] & df["errado"]).sum())
perc_accuracy_into_final_third = (
    passes_into_final_third_certs / passes_into_final_third * 100.0
    if passes_into_final_third
    else 0.0
)

passes_forward = int(df["pra_frente"].sum())
passes_backward = int(df["pra_tras"].sum())
passes_right = int(df["pra_direita"].sum())
passes_left = int(df["pra_esquerda"].sum())

perc_forward = (passes_forward / total_passes * 100.0) if total_passes else 0.0

# ==========================
# Layout: left = stats, right = map
# ==========================
col_stats, col_map = st.columns([1, 2], gap="large")

# ---- Stats (LEFT)
with col_stats:
    st.subheader("Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Passes", total_passes)
    c2.metric("Successful", passes_certs)
    c3.metric("Accuracy", f"{perc_accuracy:.1f}%")
    c4.metric("Unsuccessful", passes_errs)

    st.divider()

    st.subheader("Final Third (Entry)")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Entries", passes_into_final_third)
    c6.metric("Successful", passes_into_final_third_certs)
    c7.metric("Unsuccessful", passes_into_final_third_errs)
    c8.metric("Accuracy", f"{perc_accuracy_into_final_third:.1f}%")

    st.divider()

    st.subheader("Pass Directions")
    c9, c10 = st.columns(2)
    c9.metric("Forward", passes_forward)
    c10.metric("Forward % of Total", f"{perc_forward:.1f}%")

    c11, c12 = st.columns(2)
    c11.metric("Backward", passes_backward)
    c12.metric("Right / Left", f"{passes_right} / {passes_left}")

# ---- Map (RIGHT)
with col_map:
    st.subheader("Pass Map")

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f5f5f5", line_color="#4a4a4a")
    fig, ax = pitch.draw(figsize=(7.2, 5.0))
    fig.set_dpi(100)

    # Final third line
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.2, alpha=0.25)

    # Stronger colors (errado > progressivo > certo)
    for _, row in df.iterrows():
        if row["errado"]:
            # Weak red -> now a bit stronger
            color = (0.85, 0.20, 0.20, 0.60)
            width = 1.55
            headwidth = 2.25
            headlength = 2.25
        elif row["progressivo"]:
            # Weak blue -> now a bit stronger
            color = (0.20, 0.55, 0.98, 0.55)
            width = 1.70
            headwidth = 2.35
            headlength = 2.35
        else:
            # Light gray for completed non-progressive
            color = (0.78, 0.78, 0.78, 0.26)
            width = 1.25
            headwidth = 1.95
            headlength = 1.95

        pitch.arrows(
            row["x_start"],
            row["y_start"],
            row["x_end"],
            row["y_end"],
            color=color,
            width=width,
            headwidth=headwidth,
            headlength=headlength,
            ax=ax,
        )

    ax.set_title("Compiled Pass Map", fontsize=12)

    # Elegant legend (top-left)
    legend_elements = [
        Line2D([0], [0], color=(0.20, 0.55, 0.98, 0.55), lw=3, label="Progressive Pass"),
        Line2D([0], [0], color=(0.85, 0.20, 0.20, 0.60), lw=3, label="Unsuccessful Pass"),
        Line2D([0], [0], color=(0.78, 0.78, 0.78, 0.26), lw=3, label="Successful Pass"),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        shadow=True,
        fontsize="small",
        borderpad=0.8,
        labelspacing=0.8,
    )
    legend.get_frame().set_alpha(1.0)

    # Attack direction arrow (middle field, bottom)
    arrow = FancyArrowPatch(
        (0.45, 0.05),
        (0.55, 0.05),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2,
        color="#333333",
    )
    fig.patches.append(arrow)
    fig.text(
        0.5,
        0.02,
        "Attack Direction",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )

    fig.tight_layout()

    # Render to image with controlled width (prevents oversized layout)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)

    st.image(img, width=640)  # if still too big, reduce to 560

    plt.close(fig)

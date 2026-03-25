import re
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

st.set_page_config(layout="wide", page_title="Pass Map Dashboard")
st.title("Pass Map Dashboard")

# ==========================
# Pitch / rules configuration
# ==========================
GOAL_X = 120
GOAL_Y = 40
FINAL_THIRD_LINE_X = 80  
PROGRESSIVE_THRESHOLD = 0.75  

RAW = """
Angel vs IMG (40 Min)
0:18 – Passe certo OK
0:53 – Passe certo OK
13:14 – Passe errado OK
20:22 – Passe certo OK
25:19 – Passe certo OK
28:06 – Passe certo OK
33:34 – Pass certo OK
35:02 – pass certo OK
36:09 – pass certo OK
36:47 – pass certo OK
x = 29.25, y = 20.38
x = 23.76, y = 12.90
x = 14.78, y = 31.52
x = 32.57, y = 41.99
x = 70.64, y = 55.13
x = 75.29, y = 58.29
x = 43.04, y = 50.14
x = 30.58, y = 50.31
x = 50.86, y = 49.14
x = 45.04, y = 58.62
x = 37.56, y = 58.62
x = 55.18, y = 50.47
x = 43.54, y = 39.17
x = 57.67, y = 32.85
x = 55.35, y = 30.69
x = 37.39, y = 25.37
x = 32.90, y = 52.97
x = 41.05, y = 41.33
x = 65.82, y = 40.83
x = 48.20, y = 42.66

Angel vs Orlando (40 min)
2:56 – Pass certo OK
x = 27.58, y = 52.47
x = 13.62, y = 51.47
3:25 – Pass certo OK
x = 33.57, y = 39.83
x = 32.57, y = 25.70
4:14 – Pass certo OK
x = 50.19, y = 53.13
x = 44.87, y = 61.78
5:04 – pass certo OK
x = 53.35, y = 47.81
x = 48.20, y = 56.79
5:41 – pass errado OK
x = 54.85, y = 44.99
x = 55.18, y = 37.01
5:50 – pass certo OK
x = 54.18, y = 73.25
x = 66.15, y = 78.57
6:31 – pass certo OK
x = 22.10, y = 28.36
x = 36.56, y = 23.88
7:45 – pass certo OK
x = 59.83, y = 61.61
x = 70.31, y = 53.96
8:43 – pass certo OK
x = 31.08, y = 59.28
x = 38.22, y = 70.75
8:48 – pass certo OK
x = 51.02, y = 54.46
x = 60.66, y = 50.64
9:03 – pass certo OK
x = 55.35, y = 48.15
x = 66.48, y = 74.74
9:09 – pass certo OK
x = 59.17, y = 64.27
x = 50.69, y = 35.51
10:40 – pass certo OK
x = 58.50, y = 47.48
x = 54.35, y = 31.69
11:31 – pass certo OK
x = 78.29, y = 59.28
x = 87.59, y = 75.57
12:58 – pass certo OK
x = 52.52, y = 57.12
x = 45.70, y = 40.33
18:19 – passe certo OK
x = 28.91, y = 61.11
x = 44.37, y = 56.62
19:04 – passe certo OK
x = 46.04, y = 55.46
x = 45.21, y = 33.68
19:35 – passe certo OK
x = 69.14, y = 57.12
x = 64.16, y = 40.66
24:11 – passe certo OK
x = 29.75, y = 57.29
x = 34.57, y = 75.91
24:36 – passe certo OK
x = 29.91, y = 32.02
x = 39.39, y = 46.98
25:29 – passe certo OK
x = 90.25, y = 58.12
x = 98.57, y = 63.44
25:41 – passe certo OK
x = 94.24, y = 64.44
x = 95.41, y = 75.57
28:10 – passe certo OK
x = 26.92, y = 26.37
x = 17.11, y = 26.70
30:34 – pass certo OK
x = 58.67, y = 32.85
x = 71.30, y = 39.50
30:46 – pass certo OK
x = 76.12, y = 45.49
x = 91.25, y = 47.98
32:32 – pass certo OK
x = 102.89, y = 47.15
x = 98.40, y = 41.66
33:28 – pass certo/rec
x = 73.80, y = 71.09
x = 76.29, y = 76.07
35:07 – pass errado OK
x = 82.94, y = 48.31
x = 95.24, y = 51.14
35:44 – pass certo OK
x = 59.67, y = 48.48
x = 70.31, y = 71.92
39:33 – pass certo OK
x = 75.96, y = 37.34
x = 88.76, y = 31.85

Angel vs Weston (58:20)
0:13 – pass certo OK
1:50 – passe certo OK
6:16 – pass certo OK
8:14 – pass certo OK
15:36 – pass certo OK
18:22 – pass certo OK
24:33 – pass certo OK
26:40 – pass certo OK
28:37 – pass certo OK
29:40 – pass certo OK
29:48 – pass certo OK
30:35 – pass errado OK
34:09 – pass certo OK
36:25 – pass certo OK
38:11 – pass certo OK
38:21 – pass certo OK
40:25 – pass certo OK
43:56 – passe certo OK
46:23 – passe certo OK
46:40 – passe certo OK
48:15 – passe certo OK
53:42 – pass errado OK
54:58 – pass certo OK
57:16 – pass certo OK
x = 44.37, y = 28.03
x = 37.23, y = 53.46
x = 44.71, y = 18.72
x = 39.72, y = 10.58
x = 37.89, y = 44.99
x = 21.77, y = 40.83
x = 34.57, y = 35.84
x = 46.04, y = 36.51
x = 28.58, y = 53.13
x = 36.56, y = 61.11
x = 34.40, y = 23.88
x = 25.42, y = 35.51
x = 64.99, y = 35.68
x = 76.29, y = 11.57
x = 38.22, y = 39.83
x = 29.08, y = 23.21
x = 75.46, y = 31.02
x = 65.15, y = 45.98
x = 28.25, y = 35.84
x = 32.90, y = 20.05
x = 29.75, y = 24.71
x = 17.44, y = 18.56
x = 44.04, y = 49.64
x = 53.35, y = 58.95
x = 56.68, y = 37.34
x = 43.88, y = 34.02
x = 40.22, y = 41.50
x = 23.59, y = 50.14
x = 77.95, y = 31.36
x = 92.25, y = 48.31
x = 85.27, y = 61.28
x = 96.07, y = 66.10
x = 34.40, y = 38.67
x = 26.59, y = 24.87
x = 44.21, y = 13.90
x = 34.23, y = 16.89
x = 45.87, y = 33.68
x = 37.06, y = 19.05
x = 55.68, y = 56.12
x = 71.30, y = 73.58
x = 30.74, y = 34.35
x = 16.61, y = 27.03
x = 64.65, y = 66.76
x = 69.97, y = 73.58
x = 33.07, y = 49.48
x = 24.59, y = 55.46
x = 48.20, y = 54.13
x = 60.33, y = 54.96

Angles vs South Florida (1:03)
03:10 – pass certo OK
05:20 – pass certo OK
08:25 – pass errado OK
10:02 – pass errado OK
11:45 – pass certo OK
15:57 – pass certo OK
17:14 – pass certo OK
17:25 – pass errado OK
19:47 – pass errado OK
21:16 – pass errado OK
22:07 – pass certo OK
x = 38.89, y = 29.53
x = 30.74, y = 51.97
x = 55.18, y = 26.04
x = 67.98, y = 45.98
x = 57.84, y = 34.68
x = 71.97, y = 37.34
x = 36.89, y = 39.00
x = 58.50, y = 66.10
x = 67.15, y = 36.18
x = 75.13, y = 24.21
x = 25.76, y = 31.85
x = 11.79, y = 18.39
x = 68.31, y = 32.19
x = 54.68, y = 38.17
x = 57.17, y = 45.65
x = 69.48, y = 46.82
x = 81.11, y = 12.41
x = 83.94, y = 15.06
x = 64.99, y = 41.83
x = 69.64, y = 79.90
x = 42.05, y = 29.19
x = 46.87, y = 5.26
"""

MATCH_START_PREFIXES = ("angel vs", "angles vs")
TIME_LINE_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*(?:[-–]\s*)?", re.IGNORECASE)
COORD_RE = re.compile(
    r"x\s*=\s*([-+]?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

def parse_matches(raw_text: str) -> dict[str, pd.DataFrame]:
    lines = raw_text.splitlines()
    matches_lines: dict[str, list[str]] = {}
    current_match = None

    def normalize_match_name(line: str) -> str | None:
        lower = line.strip().lower()
        if not any(lower.startswith(p) for p in MATCH_START_PREFIXES):
            return None
        if lower.startswith("angel vs"):
            rest = line.strip()[len("Angel vs") :].strip()
            opponent = rest.split("(")[0].strip()
            return f"vs {opponent}"
        if lower.startswith("angles vs"):
            rest = line.strip()[len("Angles vs") :].strip()
            opponent = rest.split("(")[0].strip()
            return f"vs {opponent}"
        return None

    for line in lines:
        m_name = normalize_match_name(line)
        if m_name is not None:
            current_match = m_name
            matches_lines[current_match] = []
            continue
        if current_match is not None:
            matches_lines[current_match].append(line)

    out = {}
    for match_name, m_lines in matches_lines.items():
        pass_success = []
        coord_points = []
        for line in m_lines:
            ll = line.strip().lower()
            if TIME_LINE_RE.match(line):
                pass_success.append("errado" not in ll)
            cm = COORD_RE.search(line)
            if cm:
                coord_points.append((float(cm.group(1)), float(cm.group(2))))

        coord_pairs = len(coord_points) // 2
        n_passes = min(len(pass_success), coord_pairs)
        if n_passes == 0: continue

        passes = []
        for i in range(n_passes):
            start, end = coord_points[2 * i], coord_points[2 * i + 1]
            passes.append({
                "x_start": start[0], "y_start": start[1],
                "x_end": end[0], "y_end": end[1],
                "certo": pass_success[i],
                "errado": not pass_success[i]
            })

        df = pd.DataFrame(passes)
        dist_i = np.sqrt((GOAL_X - df["x_start"])**2 + (GOAL_Y - df["y_start"])**2)
        dist_f = np.sqrt((GOAL_X - df["x_end"])**2 + (GOAL_Y - df["y_end"])**2)
        df["progressive"] = dist_f <= dist_i * PROGRESSIVE_THRESHOLD
        df["into_final_third"] = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)
        df["forward"] = df["x_end"] > df["x_start"]
        df["backward"] = df["x_end"] < df["x_start"]
        df["right"] = df["y_end"] > df["y_start"]
        df["left"] = df["y_end"] < df["y_start"]
        out[match_name] = df
    return out

def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    acc = (df["certo"].sum() / total * 100.0) if total else 0.0
    
    prog_tentados = int(df["progressive"].sum())
    prog_acertados = int((df["progressive"] & df["certo"]).sum())
    
    tf_tentados = int(df["into_final_third"].sum())
    tf_acertados = int((df["into_final_third"] & df["certo"]).sum())
    
    # NOVAS MÉTRICAS DE DIRECIONAMENTO
    fwd_count = int(df["forward"].sum())
    back_count = int(df["backward"].sum())
    fwd_pct_total = (fwd_count / total * 100.0) if total else 0.0
    
    return {
        "total": total,
        "certo": int(df["certo"].sum()),
        "errado": int(df["errado"].sum()),
        "acc": acc,
        "prog_tent": prog_tentados,
        "prog_acer": prog_acertados,
        "tf_tent": tf_tentados,
        "tf_acer": tf_acertados,
        "fwd": fwd_count,
        "back": back_count,
        "fwd_total_pct": fwd_pct_total  # % de passes pra frente sobre o total
    }

def draw_pass_map(df: pd.DataFrame, title: str) -> Image.Image:
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#f5f5f5", line_color="#4a4a4a")
    fig, ax = pitch.draw(figsize=(6.8, 4.5))
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", linewidth=1.1, alpha=0.25)
    
    for _, row in df.iterrows():
        if row["errado"]:
            # Vermelho mais sólido
            color, width = (0.95, 0.18, 0.18, 0.80), 1.55 
        elif row["progressive"]:
            # Azul Progressivo bem destacado (Alpha 0.95)
            color, width = (0.15, 0.50, 1.00, 0.75), 1.80 
        else:
            # Passe Certo comum mais visível (Alpha 0.65)
            color, width = (0.60, 0.60, 0.60, 0.50), 1.25 
            
        pitch.arrows(row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                     color=color, width=width, headwidth=2, headlength=2, ax=ax)
        
    ax.set_title(title, fontsize=12, pad=10)
    
    # Legenda ajustada para ser opaca (Alpha 1.0)
    legend_elements = [
        Line2D([0], [0], color=(0.15, 0.50, 1.00, 0.95), lw=2.5, label="Progressive Pass"),
        Line2D([0], [0], color=(0.95, 0.18, 0.18, 0.85), lw=2.5, label="Unsuccessful Pass"),
        Line2D([0], [0], color=(0.60, 0.60, 0.60, 0.65), lw=2.5, label="Completed Passs"),
    ]
    ax.legend(
        handles=legend_elements, 
        loc="upper left", 
        fontsize="x-small", 
        frameon=True, 
        facecolor="white", 
        framealpha=1.0,  # Legenda sem transparência
        edgecolor="#4a4a4a"
    )
    
    arrow = FancyArrowPatch((0.45, 0.05), (0.55, 0.05), transform=fig.transFigure,
                             arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#333333")
    fig.patches.append(arrow)
    fig.text(0.5, 0.02, "Attack Direction", ha="center", fontsize=9, color="#333333")
    
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)

# --- Lógica do Dashboard ---
matches_data = parse_matches(RAW)
df_all = pd.concat(matches_data.values(), ignore_index=True)
full_data = {"All games": df_all}
full_data.update(matches_data)

st.sidebar.header("📋 Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(full_data.keys()), index=0)

st.sidebar.divider()

# O NOVO SLICER (Filtro de passes para frente)
apenas_frente = st.sidebar.checkbox("Show only forward passes", value=False)

# Pegamos os dados originais do jogo selecionado
df_sel = full_data[selected_match]

# Calculamos as estatísticas SEMPRE sobre o total do jogo (para manter o contexto)
stats = compute_stats(df_sel)

# Aplicamos o filtro APENAS no DataFrame que será enviado para o MAPA
df_mapa = df_sel.copy()
if apenas_frente:
    df_mapa = df_mapa[df_mapa["forward"] == True]

col_stats, col_map = st.columns([1, 2], gap="large")

with col_stats:
    st.subheader("📊 Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", stats["total"])
    c2.metric("Succesful", stats["certo"])
    c3.metric("%", f'{stats["acc"]:.1f}%')
    
    st.divider()
    st.subheader("🚀 Progressive Passes")
    p1, p2 = st.columns(2)
    p1.metric("Attempts", stats["prog_tent"])
    p2.metric("Successful", stats["prog_acer"])
    
    st.divider()
    st.subheader("🎯 Passes into the Final Third")
    f1, f2 = st.columns(2)
    f1.metric("Attempts", stats["tf_tent"])
    f2.metric("Sucessful", stats["tf_acer"])
    
    # SEÇÃO ALTERADA CONFORME SOLICITADO
    st.divider()
    st.subheader("↕️ Direction")
    d1, d2, d3 = st.columns(3)
    d1.metric("Forward", stats["fwd"])
    d2.metric("Backward", stats["back"])
    d3.metric("% Vertical", f'{stats["fwd_total_pct"]:.1f}%')
    st.caption("This represent the % of forward passes.")

with col_map:
    # Definimos o subtítulo com base no filtro ativo
    label_filtro = " (Only Forward)" if apenas_frente else ""
    st.subheader(f"Pass Map{label_filtro}")
    st.image(draw_pass_map(df_mapa, f"Map: {selected_match}"), width=640)

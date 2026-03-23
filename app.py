import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
from PIL import Image
import numpy as np

# ==========================
# SESSION STATE
# ==========================
if "events" not in st.session_state:
    st.session_state.events = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None

# ==========================
# TITLE
# ==========================
st.title("📊 Event Tagging Tool (StatsBomb)")

# ==========================
# INPUT TEMPO
# ==========================
tempo = st.text_input("Tempo (MM:SS)", placeholder="Ex: 10:53")

# ==========================
# DESENHAR CAMPO
# ==========================
pitch = Pitch(
    pitch_type='statsbomb',
    pitch_color='#0E1117',
    line_color='white'
)

fig, ax = pitch.draw(figsize=(8, 5))

# ==========================
# SETA DIREÇÃO
# ==========================
ax.annotate(
    '',
    xy=(110, 40), xytext=(10, 40),
    arrowprops=dict(facecolor='white', arrowstyle='->', lw=2)
)

ax.text(60, 45, "ATAQUE ➡️", color='white', ha='center', fontsize=10)

# ==========================
# PLOT ÚLTIMO CLIQUE
# ==========================
if st.session_state.last_click:
    x, y = st.session_state.last_click
    ax.scatter(x, y, color='red', s=80)

# ==========================
# FIG → IMAGE (CORRETO)
# ==========================
buf = BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
buf.seek(0)

img = Image.open(buf)
img_array = np.array(img)

height, width = img_array.shape[:2]

# ==========================
# CAPTURA CLIQUE
# ==========================
coords = streamlit_image_coordinates(img_array)

if coords is not None:
    img_x = coords["x"]
    img_y = coords["y"]

    # converter para padrão StatsBomb
    x = (img_x / width) * 120
    y = 80 - (img_y / height) * 80

    st.session_state.last_click = (round(x, 2), round(y, 2))

# ==========================
# MOSTRAR COORDENADAS
# ==========================
if st.session_state.last_click:
    st.write(f"📍 Coordenadas: {st.session_state.last_click}")

# ==========================
# FUNÇÃO ADICIONAR EVENTO
# ==========================
def add_event(event):
    if not tempo:
        st.warning("Digite o tempo antes de salvar.")
        return
    
    if not st.session_state.last_click:
        st.warning("Clique no campo antes de salvar.")
        return
    
    x, y = st.session_state.last_click

    st.session_state.events.append({
        "tempo": tempo,
        "evento": event,
        "x": x,
        "y": y
    })

# ==========================
# BOTÕES DE EVENTO
# ==========================
st.subheader("Eventos")

col1, col2 = st.columns(2)

with col1:
    if st.button("Passe Certo"):
        add_event("Passe_Certo")

    if st.button("Passe Longo Certo"):
        add_event("Passe_Longo_Certo")

with col2:
    if st.button("Passe Errado"):
        add_event("Passe_Errado")

    if st.button("Passe Longo Errado"):
        add_event("Passe_Longo_Errado")

# ==========================
# TABELA
# ==========================
st.subheader("Eventos registrados")

df = pd.DataFrame(st.session_state.events)
st.dataframe(df, use_container_width=True)

# ==========================
# DOWNLOAD CSV
# ==========================
if not df.empty:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Baixar CSV",
        csv,
        "eventos_statsbomb.csv",
        "text/csv"
    )

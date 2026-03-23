import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# ==========================
# SESSION STATE
# ==========================
if "events" not in st.session_state:
    st.session_state.events = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None

# ==========================
# TÍTULO
# ==========================
st.title("📊 Event Tagging Tool")

# ==========================
# INPUT DE TEMPO
# ==========================
tempo = st.text_input("Tempo do evento (MM:SS)", placeholder="Ex: 10:53")

# ==========================
# DESENHAR CAMPO
# ==========================
pitch = Pitch(pitch_type='opta', pitch_color='grass', line_color='white')

fig, ax = pitch.draw(figsize=(6, 4))

# Plotar último clique
if st.session_state.last_click:
    x, y = st.session_state.last_click
    ax.scatter(x, y, color='red', s=100)

# Captura clique
click = st.pyplot(fig, clear_figure=False)

# ==========================
# CAPTURA DE COORDENADA
# ==========================
# OBS: Streamlit puro não captura clique diretamente,
# então usamos workaround com coordenadas simuladas
# (na prática você pode integrar streamlit-image-coordinates)

from streamlit_image_coordinates import streamlit_image_coordinates

coords = streamlit_image_coordinates(fig)

if coords is not None:
    st.session_state.last_click = (coords["x"], coords["y"])

# Mostrar coordenadas
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
# BOTÕES
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
# DOWNLOAD
# ==========================
if not df.empty:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Baixar CSV",
        csv,
        "eventos.csv",
        "text/csv"
    )

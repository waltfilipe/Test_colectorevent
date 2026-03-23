import streamlit as st
import pandas as pd

# ==========================
# SESSION STATE
# ==========================
if "events" not in st.session_state:
    st.session_state.events = []

# ==========================
# FUNÇÃO FORMATAR TEMPO
# ==========================
def format_time(value):
    value = value.strip()

    if not value.isdigit():
        return None

    value = value.zfill(4)  # garante 4 dígitos

    minutes = value[:-2]
    seconds = value[-2:]

    return f"{int(minutes):02d}:{int(seconds):02d}"

# ==========================
# TITLE
# ==========================
st.title("📊 Event Tagging Tool (Rápido)")

# ==========================
# INPUT TEMPO
# ==========================
raw_time = st.text_input("Tempo (somente números)", placeholder="Ex: 1053 → 10:53")

formatted_time = format_time(raw_time)

if formatted_time:
    st.success(f"Tempo: {formatted_time}")
elif raw_time:
    st.error("Digite apenas números")

# ==========================
# FUNÇÃO ADICIONAR EVENTO
# ==========================
def add_event(event):
    if not formatted_time:
        st.warning("Digite um tempo válido")
        return

    st.session_state.events.append({
        "tempo": formatted_time,
        "evento": event
    })

# ==========================
# BOTÕES DE EVENTO
# ==========================
st.subheader("Eventos")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Passe Certo"):
        add_event("Passe_Certo")

    if st.button("Passe Errado"):
        add_event("Passe_Errado")

with col2:
    if st.button("Duelo Vencido"):
        add_event("Duelo_Vencido")

    if st.button("Duelo Perdido"):
        add_event("Duelo_Perdido")

with col3:
    if st.button("Finalização"):
        add_event("Finalizacao")

    if st.button("Outro"):
        add_event("Outro")

# ==========================
# BOTÃO CLEAR
# ==========================
st.markdown("---")

if st.button("🧹 Limpar Eventos"):
    st.session_state.events = []
    st.success("Eventos apagados")

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
        "eventos.csv",
        "text/csv"
    )

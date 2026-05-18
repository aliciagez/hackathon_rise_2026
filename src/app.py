import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="SMHI Temperaturanalys", layout="wide")
st.title("🌡️ SMHI Temperaturanalys & Prediktion")

st.sidebar.header("Data")

st.sidebar.subheader("Välj datum för prediktion")

valt_år = st.sidebar.selectbox("År", options=[2027, 2028], index=0)

valt_månad = st.sidebar.selectbox(
    "Månad", options=list(range(1, 13)), format_func=lambda m: [
        "Jan","Feb","Mar","Apr","Maj","Jun",
        "Jul","Aug","Sep","Okt","Nov","Dec"
    ][m - 1]
)

# RIKTIG DATA
df = pd.read_csv("clean_data/cleand_smhi.csv")
df["datum"] = pd.to_datetime(df["Representativt dygn"])
df = df.set_index("datum")
df["temperatur"] = df["Lufttemperatur"]
månadsdata = df["temperatur"].resample("ME").mean()

@st.cache_data
def kör_sarima(data):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=48)
    return forecast.predicted_mean, forecast.conf_int()

valt_datum = f"{valt_år}-{valt_månad:02d}"

st.info("Tränar SARIMA-modellen... detta tar ~30 sekunder")
prediktion, ki = kör_sarima(månadsdata)

if valt_datum in prediktion.index.strftime("%Y-%m"):
    mask = prediktion.index.strftime("%Y-%m") == valt_datum
    temp = prediktion[mask].values[0]
    st.sidebar.metric(
        label=f"Prediktion {valt_datum}",
        value=f"{temp:.1f}°C"
    )
else:
    st.sidebar.warning("Inget prediktionsvärde för det datumet.")

col1, col2, col3 = st.columns(3)
col1.metric("Prediktion 2027", f"{prediktion['2027'].mean():.1f}°C")
col2.metric("Prediktion 2028", f"{prediktion['2028'].mean():.1f}°C")
col3.metric("Felmarginal", f"±{(ki.iloc[:, 1] - ki.iloc[:, 0]).mean() / 2:.1f}°C")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=månadsdata.index,
        y=månadsdata,
        name="Historisk data",
        line=dict(color="#1f77b4"),
    )
)

fig.add_trace(
    go.Scatter(
        x=prediktion.index,
        y=prediktion,
        name="Prediktion",
        line=dict(color="red", dash="dash"),
    )
)

fig.add_trace(
    go.Scatter(
        x=list(ki.index) + list(ki.index[::-1]),
        y=list(ki.iloc[:, 0]) + list(ki.iloc[:, 1][::-1]),
        fill="toself",
        fillcolor="rgba(255,0,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Felmarginal",
    )
)

fig.update_layout(
    title="Temperaturutveckling & Prediktion",
    xaxis_title="År",
    yaxis_title="Temperatur (°C)",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import calendar
from datetime import date

st.set_page_config(page_title="SMHI Temperaturanalys", layout="wide")
st.title("🌡️ SMHI Temperaturanalys & Prediktion")

st.sidebar.header("Data")
st.sidebar.subheader("Välj datum för prediktion")

idag = date.today()
valt_år = st.sidebar.selectbox(
    "År",
    options=list(range(idag.year, idag.year + 5)),
    index=0
)
valt_månad = st.sidebar.selectbox(
    "Månad",
    options=list(range(1, 13)),
    format_func=lambda m: [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Maj",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Okt",
        "Nov",
        "Dec",
    ][m - 1],
)
dagar_i_månad = calendar.monthrange(valt_år, valt_månad)[1]
valt_dag = st.sidebar.selectbox(
    "Dag",
    options=list(range(1, dagar_i_månad + 1))
)


# RIKTIG DATA
df = pd.read_csv("clean_data/cleand_smhi.csv")
df["datum"] = pd.to_datetime(df["Representativt dygn"])
df = df.set_index("datum")
df["temperatur"] = df["Lufttemperatur"]
månadsdata = df["temperatur"].resample("ME").mean()

# TRAIN/TEST SPLIT — 80% träning, 20% test
split = int(len(månadsdata) * 0.8)
train = månadsdata[:split]
test = månadsdata[split:]


@st.cache_data
def kör_sarima(train, test):
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit(disp=False)
    test_pred = result.get_forecast(steps=len(test))
    test_medel = test_pred.predicted_mean
    forecast = result.get_forecast(steps=len(test) + 48)
    prediktion = forecast.predicted_mean.iloc[len(test) :]
    ki = forecast.conf_int().iloc[len(test) :]
    return test_medel, prediktion, ki


def månads_till_dagar(prediktion, df):
    daglig_std = df["temperatur"].groupby(df.index.month).std()
    dagspred = []
    dagsdatum = []
    for datum, medel in prediktion.items():
        std = daglig_std[datum.month]
        antal_dagar = datum.days_in_month
        dagar = np.random.normal(loc=medel, scale=std, size=antal_dagar)
        start = datum.replace(day=1)
        datum_lista = pd.date_range(start, periods=antal_dagar, freq="D")
        dagspred.extend(dagar)
        dagsdatum.extend(datum_lista)
    return pd.Series(dagspred, index=dagsdatum)


valt_datum = f"{valt_år}-{valt_månad:02d}"


test_pred, prediktion, ki = kör_sarima(train, test)
dagspred = månads_till_dagar(prediktion, df)

if valt_datum in prediktion.index.strftime("%Y-%m"):
    mask = prediktion.index.strftime("%Y-%m") == valt_datum
    temp_månad = prediktion[mask].values[0]
    lower = ki.iloc[:, 0][mask].values[0]
    upper = ki.iloc[:, 1][mask].values[0]

    st.sidebar.metric("Månadssnitt", f"{temp_månad:.1f}°C")
    st.sidebar.caption(f"Felmarginal: {lower:.1f}°C – {upper:.1f}°C")

    valt_ts = pd.Timestamp(f"{valt_år}-{valt_månad:02d}-{valt_dag:02d}")
    if valt_ts in dagspred.index:
        temp_dag = dagspred.loc[valt_ts]
        daglig_std = df["temperatur"].groupby(df.index.month).std()
        std = daglig_std[valt_månad]
        månads_osäkerhet = (upper - lower) / 2
        lower_dag = temp_dag - månads_osäkerhet - std
        upper_dag = temp_dag + månads_osäkerhet + std
        st.sidebar.metric(f"Dag {valt_dag}", f"{temp_dag:.1f}°C")
        st.sidebar.caption(f"Felmarginal: {lower_dag:.1f}°C – {upper_dag:.1f}°C")
else:
    st.sidebar.warning("Inget prediktionsvärde för det datumet.")


# MÄTVÄRDEN
mae = mean_absolute_error(test, test_pred)
rmse = np.sqrt(mean_squared_error(test, test_pred))

# METRICS
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prediktion 2027", f"{prediktion['2027'].mean():.1f}°C")
col2.metric("Prediktion 2028", f"{prediktion['2028'].mean():.1f}°C")
col3.metric("Felmarginal", f"±{(ki.iloc[:, 1] - ki.iloc[:, 0]).mean() / 2:.1f}°C")
col4.metric("MAE", f"{mae:.2f}°C")

# MÅNADSDIAGRAM
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=train.index, y=train, name="Träningsdata", line=dict(color="#1f77b4"))
)

fig.add_trace(
    go.Scatter(x=test.index, y=test, name="Testdata", line=dict(color="orange"))
)

fig.add_trace(
    go.Scatter(
        x=test_pred.index,
        y=test_pred,
        name="Modell på testdata",
        line=dict(color="green", dash="dash"),
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

# DAGSPREDIKTION
st.subheader("📅 Dagsprediktion")

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=dagspred.index,
        y=dagspred,
        name="Dagsprediktion",
        line=dict(color="red", width=1),
    )
)

fig2.update_layout(
    title="Daglig temperaturprediktion 2026-2030",
    xaxis_title="Datum",
    yaxis_title="Temperatur (°C)",
    hovermode="x unified",
)

st.plotly_chart(fig2, use_container_width=True)

# MODELLPRESTANDA TABELL
st.subheader("📊 Modellens prestanda")
st.table(
    {
        "Mätvärde": ["MAE", "RMSE"],
        "Värde": [f"{mae:.2f}°C", f"{rmse:.2f}°C"],
        "Förklaring": ["Snittfel i grader", "Straffar stora fel mer"],
    }
)

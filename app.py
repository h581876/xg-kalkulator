import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.features import FEATURES, add_features

st.title("xG-kalkulator")
st.write("Hvor sannsynlig er det at skuddet blir mål?")


@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


model = load_model()

col1, col2 = st.columns(2)
with col1:
    x = st.slider("X-posisjon (120 = mållinjen)", 60, 120, 108)
    y = st.slider("Y-posisjon (40 = midt for mål)", 0, 80, 40)
    body = st.selectbox("Kroppsdel", ["Fot", "Hode"])
    play = st.selectbox("Situasjon", ["Åpent spill", "Dødball (frispark/hjørnespark)"])
    pressure = st.checkbox("Under press")
    first_time = st.checkbox("Første berøring")

row = add_features(pd.DataFrame([{
    "x": x, "y": y,
    "is_header": int(body == "Hode"),
    "set_piece": int(play != "Åpent spill"),
    "under_pressure": int(pressure),
    "first_time": int(first_time)}]))
p = model.predict_proba(row[FEATURES])[0, 1]

with col2:
    st.metric("Sannsynlighet for mål", f"{p:.1%}")
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], "k")          # banen
    ax.plot([102, 120, 120, 102, 102], [18, 18, 62, 62, 18], "k")    # 16-meteren
    ax.plot([120, 120], [36, 44], "r", lw=5)                         # målet
    ax.scatter([x], [y], s=120, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.caption("Data source: StatsBomb Open Data")
    st.image(
        "assets/hudl-statsbomb-logo-default.png",
        width=150
    )

    st.markdown(
        "[StatsBomb Open Data](https://github.com/hudl/open-data)"
    )
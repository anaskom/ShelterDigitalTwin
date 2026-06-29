import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Shelter Digital Twin", layout="wide")

st.title("Digital Twin of School Emergency Shelter")

# --- Sidebar controls ---
st.sidebar.header("Inputs")
people = st.sidebar.slider("Number of people", 5, 100, 45)
outside_temp = st.sidebar.slider("Outdoor temperature, °C", -10, 35, 12)
simulation_speed = st.sidebar.slider("Simulation speed", 0.1, 2.0, 0.5)

# --- Initial state ---
if "co2" not in st.session_state:
    st.session_state.co2 = 700
if "history" not in st.session_state:
    st.session_state.history = []
if "fan_on" not in st.session_state:
    st.session_state.fan_on = False

# --- Digital twin logic ---
co2_threshold = 1000

# CO2 grows depending on number of people
co2_growth = people * 0.7

# If CO2 is too high, system automatically turns ventilation ON
if st.session_state.co2 > co2_threshold:
    st.session_state.fan_on = True
elif st.session_state.co2 < 800:
    st.session_state.fan_on = False

# Ventilation decreases CO2
ventilation_effect = 55 if st.session_state.fan_on else 10

# Update CO2
st.session_state.co2 += co2_growth - ventilation_effect
st.session_state.co2 = max(420, st.session_state.co2)

# Other simplified parameters
indoor_temp = 21 + people * 0.02 - (2 if st.session_state.fan_on else 0)
humidity = 45 + people * 0.1
ach = 4.5 if st.session_state.fan_on else 1.2

st.session_state.history.append({
    "Time": len(st.session_state.history),
    "CO2": st.session_state.co2,
    "Temperature": indoor_temp,
    "Humidity": humidity,
    "ACH": ach,
    "Ventilation": "ON" if st.session_state.fan_on else "OFF"
})

df = pd.DataFrame(st.session_state.history[-60:])

# --- Layout ---
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Digital Shelter State")

    if st.session_state.co2 < 800:
        room_color = "#8BC34A"
        status = "Safe"
    elif st.session_state.co2 < 1000:
        room_color = "#FFC107"
        status = "Warning"
    else:
        room_color = "#F44336"
        status = "Critical"

    fig_room = go.Figure()

    # Room rectangle
    fig_room.add_shape(
        type="rect",
        x0=0, y0=0, x1=10, y1=6,
        line=dict(color="black", width=3),
        fillcolor=room_color,
        opacity=0.55
    )

    # People dots
    np.random.seed(1)
    max_dots = min(people, 80)
    x = np.random.uniform(1, 9, max_dots)
    y = np.random.uniform(1, 5, max_dots)
    fig_room.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=8, color="black"),
        name="People"
    ))

    # Ventilation
    vent_color = "blue" if st.session_state.fan_on else "gray"
    fig_room.add_shape(
        type="rect",
        x0=9.5, y0=2.3, x1=10.4, y1=3.7,
        line=dict(color=vent_color, width=3),
        fillcolor=vent_color,
        opacity=0.7
    )

    fig_room.add_annotation(
        x=5, y=6.5,
        text=f"Status: {status} | Ventilation: {'ON' if st.session_state.fan_on else 'OFF'}",
        showarrow=False,
        font=dict(size=18)
    )

    fig_room.update_layout(
        height=450,
        xaxis=dict(visible=False, range=[-0.5, 11]),
        yaxis=dict(visible=False, range=[-0.5, 7]),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_room, use_container_width=True)

with col2:
    st.subheader("Current values")
    st.metric("CO₂", f"{st.session_state.co2:.0f} ppm")
    st.metric("Indoor temperature", f"{indoor_temp:.1f} °C")
    st.metric("Humidity", f"{humidity:.1f} %")
    st.metric("Air Change Rate", f"{ach:.1f} ACH")
    st.metric("Automatic action", "Ventilation ON" if st.session_state.fan_on else "No action")

st.subheader("Real-time CO₂ dynamics")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Time"],
    y=df["CO2"],
    mode="lines+markers",
    name="CO₂"
))

fig.add_hline(y=1000, line_dash="dash", annotation_text="CO₂ threshold")
fig.update_layout(
    height=350,
    xaxis_title="Simulation step",
    yaxis_title="CO₂, ppm"
)

st.plotly_chart(fig, use_container_width=True)

time.sleep(simulation_speed)
st.rerun()
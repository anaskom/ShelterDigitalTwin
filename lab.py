import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Adaptive Coworking Digital Twin",
    layout="wide"
)

# -----------------------------
# INITIAL STATE
# -----------------------------

def init_state(force=False):
    defaults = {
        "step": 0,
        "co2": 650.0,
        "temperature": 22.0,
        "humidity": 45.0,
        "pm25": 8.0,
        "pm10": 14.0,
        "tvoc": 250.0,
        "formaldehyde": 18.0,
        "co": 0.5,
        "oxygen": 20.9,
        "noise": 42.0,
        "light": 450.0,
        "battery": 100.0,
        "energy": 1.2,
        "history": []
    }

    for key, value in defaults.items():
        if force or key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


init_state()


# -----------------------------
# MODE CONFIGURATION
# -----------------------------

def get_config(mode):
    if mode == "Coworking mode":
        return {
            "goal": "Maximize comfort and productivity while saving energy",
            "co2_warn": 1000,
            "co2_crit": 1500,
            "temp_min": 20,
            "temp_max": 24,
            "humidity_min": 40,
            "humidity_max": 60,
            "lux_target": 500,
            "noise_warn": 55,
            "pm25_warn": 15,
            "tvoc_warn": 500,
            "hcho_warn": 50,
            "vent_base": 25,
            "vent_high": 75,
            "battery_save": 30,
            "outdoor_pm25_high": 25,
            "glare_lux": 1000
        }
    else:
        return {
            "goal": "Maintain healthy conditions and extend autonomous operation time",
            "co2_warn": 1500,
            "co2_crit": 2200,
            "temp_min": 16,
            "temp_max": 28,
            "humidity_min": 30,
            "humidity_max": 70,
            "lux_target": 100,
            "noise_warn": 75,
            "pm25_warn": 25,
            "tvoc_warn": 700,
            "hcho_warn": 80,
            "vent_base": 15,
            "vent_high": 55,
            "battery_save": 50,
            "outdoor_pm25_high": 25,
            "glare_lux": 1200
        }


# -----------------------------
# CONTROL LOGIC
# -----------------------------

def decide_actions(
    mode,
    config,
    people,
    capacity,
    room_area,
    outside_temp,
    outdoor_pm25,
    daylight,
    smoke_detected,
    water_leak
):
    s = st.session_state

    occupancy_ratio = people / capacity if capacity > 0 else 0
    density = room_area / people if people > 0 else np.inf

    # Default actions
    ventilation = 5 if people == 0 else config["vent_base"]
    outside_air_intake = ventilation
    filtration = 20
    heating = 0
    cooling = 0
    lighting = 0
    blinds = 20
    sound_masking = 0
    emergency_lights = 0
    smoke_exhaust = 0

    # CO2 and occupancy control
    if s.co2 > config["co2_warn"] or occupancy_ratio > 0.8:
        ventilation = max(ventilation, config["vent_high"])

    if s.co2 > config["co2_crit"]:
        ventilation = 100

    # Pollution control
    if s.pm25 > config["pm25_warn"] or s.pm10 > 45:
        filtration = 80

    if s.tvoc > config["tvoc_warn"] or s.formaldehyde > config["hcho_warn"]:
        filtration = 100
        ventilation = max(ventilation, 70)

    if s.co > 9 or s.oxygen < 19.5:
        ventilation = 100
        filtration = 100

    # If outdoor air is polluted, limit outside air intake and rely more on filtration
    if outdoor_pm25 > config["outdoor_pm25_high"] and s.co2 < config["co2_crit"]:
        outside_air_intake = min(ventilation, 35)
        filtration = 100
    else:
        outside_air_intake = ventilation

    # Temperature control
    hvac_mode = "Standby"

    if s.temperature < config["temp_min"]:
        heating = min(100, (config["temp_min"] - s.temperature) * 30)
        hvac_mode = "Heating"

    elif s.temperature > config["temp_max"]:
        cooling = min(100, (s.temperature - config["temp_max"]) * 30)
        hvac_mode = "Cooling"

    # In emergency mode, temperature control is less aggressive to save energy
    if mode == "Emergency shelter mode":
        heating *= 0.5
        cooling *= 0.5

    # Lighting control
    target_lux = config["lux_target"]

    if people == 0:
        target_lux = 30

    if mode == "Emergency shelter mode" and s.battery < 30:
        target_lux = 50

    artificial_needed = max(0, target_lux - daylight)
    lighting = min(100, artificial_needed / 600 * 100)

    # Blinds control
    if daylight > config["glare_lux"]:
        blinds = 80

    if s.temperature > config["temp_max"] and outside_temp > s.temperature:
        blinds = 90

    # Noise control
    if mode == "Coworking mode" and s.noise > config["noise_warn"]:
        sound_masking = 40

    # Emergency mode energy saving
    energy_saving = False

    if mode == "Emergency shelter mode" or s.battery < config["battery_save"] or people == 0:
        energy_saving = True

    if energy_saving:
        lighting *= 0.7
        sound_masking *= 0.5

    # Safety scenarios
    alarm = "OFF"
    exits = "Normal access"
    outlets = "ON"

    if mode == "Emergency shelter mode":
        emergency_lights = 60
        exits = "Emergency exits unlocked"

    if smoke_detected:
        alarm = "FIRE ALARM"
        ventilation = 0
        outside_air_intake = 0
        smoke_exhaust = 100
        filtration = 100
        emergency_lights = 100
        exits = "Emergency exits unlocked"

    if water_leak:
        outlets = "OFF in affected zone"

    return {
        "ventilation": ventilation,
        "outside_air_intake": outside_air_intake,
        "filtration": filtration,
        "heating": heating,
        "cooling": cooling,
        "hvac_mode": hvac_mode,
        "lighting": lighting,
        "blinds": blinds,
        "sound_masking": sound_masking,
        "emergency_lights": emergency_lights,
        "smoke_exhaust": smoke_exhaust,
        "energy_saving": energy_saving,
        "alarm": alarm,
        "exits": exits,
        "outlets": outlets,
        "occupancy_ratio": occupancy_ratio,
        "density": density
    }


# -----------------------------
# PHYSICAL MODEL SIMULATION
# -----------------------------

def update_physical_state(
    mode,
    actions,
    people,
    outside_temp,
    outdoor_pm25,
    daylight,
    smoke_detected,
    backup_power_only
):
    s = st.session_state

    ventilation = actions["ventilation"]
    outside_air_intake = actions["outside_air_intake"]
    filtration = actions["filtration"]
    heating = actions["heating"]
    cooling = actions["cooling"]
    lighting = actions["lighting"]
    blinds = actions["blinds"]
    sound_masking = actions["sound_masking"]
    smoke_exhaust = actions["smoke_exhaust"]

    # CO2 dynamics
    co2_generation = people * 0.55
    co2_removal = ventilation * 0.32
    s.co2 += co2_generation - co2_removal + np.random.normal(0, 5)
    s.co2 = float(np.clip(s.co2, 420, 3500))

    # Temperature dynamics
    outdoor_influence = (outside_temp - s.temperature) * (outside_air_intake / 100) * 0.025
    internal_heat = people * 0.004
    hvac_effect = heating * 0.025 - cooling * 0.035

    s.temperature += outdoor_influence + internal_heat + hvac_effect + np.random.normal(0, 0.05)
    s.temperature = float(np.clip(s.temperature, 5, 40))

    # Humidity dynamics
    s.humidity += people * 0.008 - ventilation * 0.018 + np.random.normal(0, 0.2)
    s.humidity = float(np.clip(s.humidity, 20, 85))

    # PM2.5 and PM10 dynamics
    outdoor_pm_influence = (outdoor_pm25 - s.pm25) * (outside_air_intake / 100) * 0.05
    filtration_effect = filtration * 0.07

    smoke_source = 120 if smoke_detected else 0

    s.pm25 += outdoor_pm_influence - filtration_effect + smoke_source + np.random.normal(0, 0.5)
    s.pm25 = float(np.clip(s.pm25, 0, 300))

    s.pm10 = float(np.clip(s.pm25 * 1.6 + np.random.normal(0, 1), 0, 500))

    # VOC and formaldehyde dynamics
    s.tvoc += 1.2 + people * 0.03 - ventilation * 0.18 - filtration * 0.10 + np.random.normal(0, 2)
    s.tvoc = float(np.clip(s.tvoc, 50, 1500))

    s.formaldehyde += 0.2 - ventilation * 0.03 - filtration * 0.02 + np.random.normal(0, 0.3)
    s.formaldehyde = float(np.clip(s.formaldehyde, 2, 150))

    # CO and oxygen dynamics
    if smoke_detected:
        s.co += 4
    else:
        s.co -= ventilation * 0.015

    s.co = float(np.clip(s.co, 0, 80))

    s.oxygen = 20.9 - max(s.co2 - 420, 0) * 0.0003
    s.oxygen = float(np.clip(s.oxygen, 18.0, 21.0))

    # Noise dynamics
    s.noise = 32 + people * 0.35 + np.random.normal(0, 2)
    s.noise = float(np.clip(s.noise, 25, 90))

    # Light dynamics
    daylight_after_blinds = daylight * (1 - blinds / 100 * 0.75)
    artificial_light = lighting / 100 * 650
    emergency_light = actions["emergency_lights"] / 100 * 120

    s.light = daylight_after_blinds + artificial_light + emergency_light
    s.light = float(np.clip(s.light, 0, 1500))

    # Energy use
    hvac_load = max(heating, cooling)

    s.energy = (
        0.25
        + ventilation * 0.018
        + filtration * 0.010
        + lighting * 0.006
        + hvac_load * 0.025
        + sound_masking * 0.002
        + smoke_exhaust * 0.020
    )

    if people == 0 and mode == "Coworking mode":
        s.energy *= 0.45

    s.energy = float(np.clip(s.energy, 0.1, 8))

    # Battery
    if backup_power_only:
        s.battery -= s.energy * 0.06
    else:
        s.battery += 0.03

    s.battery = float(np.clip(s.battery, 0, 100))

    # Save history
    s.history.append({
        "Step": s.step,
        "Mode": mode,
        "People": people,
        "CO2": s.co2,
        "Temperature": s.temperature,
        "Humidity": s.humidity,
        "PM2.5": s.pm25,
        "PM10": s.pm10,
        "TVOC": s.tvoc,
        "Formaldehyde": s.formaldehyde,
        "CO": s.co,
        "Oxygen": s.oxygen,
        "Noise": s.noise,
        "Light": s.light,
        "Battery": s.battery,
        "Energy": s.energy,
        "Ventilation": ventilation,
        "Filtration": filtration,
        "Lighting": lighting,
        "Heating": heating,
        "Cooling": cooling
    })

    s.step += 1

    if len(s.history) > 300:
        s.history = s.history[-300:]


# -----------------------------
# STATUS CLASSIFICATION
# -----------------------------

def classify_status(config, actions, smoke_detected, water_leak):
    s = st.session_state

    critical = (
        smoke_detected
        or s.co2 > config["co2_crit"]
        or s.co > 30
        or s.oxygen < 19.5
        or s.pm25 > 50
        or s.tvoc > 1000
        or s.battery < 10
    )

    warning = (
        s.co2 > config["co2_warn"]
        or s.temperature < config["temp_min"]
        or s.temperature > config["temp_max"]
        or s.humidity < config["humidity_min"]
        or s.humidity > config["humidity_max"]
        or s.pm25 > config["pm25_warn"]
        or s.tvoc > config["tvoc_warn"]
        or s.noise > config["noise_warn"]
        or water_leak
        or actions["occupancy_ratio"] > 0.8
    )

    if critical:
        return "Critical", "#F44336"
    elif warning:
        return "Warning", "#FFC107"
    else:
        return "Optimal", "#4CAF50"


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Digital Twin Controls")

mode = st.sidebar.radio(
    "System mode",
    ["Coworking mode", "Emergency shelter mode"]
)

config = get_config(mode)

st.sidebar.info(f"Goal: {config['goal']}")

people = st.sidebar.slider("Number of people", 0, 120, 35)
capacity = st.sidebar.slider("Design capacity", 10, 150, 60)
room_area = st.sidebar.slider("Room area, m²", 40, 300, 150)

outside_temp = st.sidebar.slider("Outdoor temperature, °C", -15, 40, 12)
outdoor_pm25 = st.sidebar.slider("Outdoor PM2.5, μg/m³", 0, 120, 8)
daylight = st.sidebar.slider("Natural daylight, lux", 0, 1500, 350)

backup_power_only = st.sidebar.checkbox(
    "Backup power only",
    value=True if mode == "Emergency shelter mode" else False
)

smoke_detected = st.sidebar.checkbox("Smoke detected", value=False)
water_leak = st.sidebar.checkbox("Water leak detected", value=False)

auto_run = st.sidebar.checkbox("Auto refresh", value=True)
refresh_speed = st.sidebar.slider("Refresh speed, seconds", 0.2, 2.0, 0.8)

if st.sidebar.button("Reset simulation"):
    init_state(force=True)
    st.rerun()


# -----------------------------
# RUN ONE SIMULATION STEP
# -----------------------------

actions = decide_actions(
    mode,
    config,
    people,
    capacity,
    room_area,
    outside_temp,
    outdoor_pm25,
    daylight,
    smoke_detected,
    water_leak
)

update_physical_state(
    mode,
    actions,
    people,
    outside_temp,
    outdoor_pm25,
    daylight,
    smoke_detected,
    backup_power_only
)

actions = decide_actions(
    mode,
    config,
    people,
    capacity,
    room_area,
    outside_temp,
    outdoor_pm25,
    daylight,
    smoke_detected,
    water_leak
)

status, status_color = classify_status(config, actions, smoke_detected, water_leak)


# -----------------------------
# MAIN DASHBOARD
# -----------------------------

st.title("Adaptive Student Coworking Digital Twin")

st.markdown(
    f"""
    **Current mode:** `{mode}`  
    **Control goal:** {config["goal"]}  
    **System status:** <span style="color:{status_color}; font-weight:bold">{status}</span>
    """,
    unsafe_allow_html=True
)

metric_cols = st.columns(6)

with metric_cols[0]:
    st.metric("People", people)

with metric_cols[1]:
    st.metric("CO₂", f"{st.session_state.co2:.0f} ppm")

with metric_cols[2]:
    st.metric("Temperature", f"{st.session_state.temperature:.1f} °C")

with metric_cols[3]:
    st.metric("PM2.5", f"{st.session_state.pm25:.1f} μg/m³")

with metric_cols[4]:
    st.metric("Battery", f"{st.session_state.battery:.1f} %")

with metric_cols[5]:
    st.metric("Energy use", f"{st.session_state.energy:.2f} kW")


# -----------------------------
# ROOM VISUALIZATION
# -----------------------------

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Live room state")

    fig_room = go.Figure()

    fig_room.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=12,
        y1=7,
        line=dict(color="black", width=3),
        fillcolor=status_color,
        opacity=0.45
    )

    # People dots
    np.random.seed(4)
    dots = min(people, 120)

    if dots > 0:
        x = np.random.uniform(1, 11, dots)
        y = np.random.uniform(1, 6, dots)

        fig_room.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=8, color="black"),
            name="People"
        ))

    # Ventilation block
    vent_color = "#2196F3" if actions["ventilation"] > 0 else "#9E9E9E"

    fig_room.add_shape(
        type="rect",
        x0=11.5,
        y0=2.5,
        x1=12.5,
        y1=4.5,
        line=dict(color=vent_color, width=3),
        fillcolor=vent_color,
        opacity=0.8
    )

    fig_room.add_annotation(
        x=6,
        y=7.5,
        text=f"Status: {status} | Ventilation: {actions['ventilation']:.0f}% | Filtration: {actions['filtration']:.0f}%",
        showarrow=False,
        font=dict(size=17)
    )

    fig_room.add_annotation(
        x=12.7,
        y=3.5,
        text="Vent",
        showarrow=False,
        font=dict(size=13)
    )

    fig_room.update_layout(
        height=460,
        xaxis=dict(visible=False, range=[-0.5, 13.5]),
        yaxis=dict(visible=False, range=[-0.5, 8]),
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False
    )

    st.plotly_chart(fig_room, use_container_width=True)


# -----------------------------
# ACTION PANEL
# -----------------------------

def progress_action(label, value, description=""):
    value = int(np.clip(value, 0, 100))
    st.write(f"**{label}: {value}%**")
    st.progress(value)
    if description:
        st.caption(description)


with right:
    st.subheader("Automatic system actions")

    progress_action(
        "Ventilation",
        actions["ventilation"],
        "Controls CO₂, humidity and general air exchange."
    )

    progress_action(
        "Outside air intake",
        actions["outside_air_intake"],
        "Reduced when outdoor PM2.5 is high."
    )

    progress_action(
        "Air filtration",
        actions["filtration"],
        "Activated for PM2.5, PM10, VOC and formaldehyde."
    )

    progress_action(
        "Lighting",
        actions["lighting"],
        "Keeps working light level in coworking mode and minimum safe light in emergency mode."
    )

    progress_action(
        "Blinds",
        actions["blinds"],
        "Controls glare and overheating from sunlight."
    )

    progress_action(
        "Heating",
        actions["heating"]
    )

    progress_action(
        "Cooling",
        actions["cooling"]
    )

    progress_action(
        "Sound masking",
        actions["sound_masking"],
        "Used only in coworking mode when noise is too high."
    )

    if actions["smoke_exhaust"] > 0:
        progress_action("Smoke exhaust", actions["smoke_exhaust"])

    st.write(f"**HVAC mode:** {actions['hvac_mode']}")
    st.write(f"**Energy saving mode:** {'ON' if actions['energy_saving'] else 'OFF'}")
    st.write(f"**Alarm:** {actions['alarm']}")
    st.write(f"**Exits:** {actions['exits']}")
    st.write(f"**Power outlets:** {actions['outlets']}")


# -----------------------------
# SENSOR DATA
# -----------------------------

st.subheader("Current sensor values")

sensor_df = pd.DataFrame({
    "Indicator": [
        "CO₂",
        "Temperature",
        "Humidity",
        "PM2.5",
        "PM10",
        "TVOC",
        "Formaldehyde",
        "Carbon monoxide",
        "Oxygen",
        "Noise",
        "Light",
        "Occupancy density"
    ],
    "Current value": [
        f"{st.session_state.co2:.0f} ppm",
        f"{st.session_state.temperature:.1f} °C",
        f"{st.session_state.humidity:.1f} %",
        f"{st.session_state.pm25:.1f} μg/m³",
        f"{st.session_state.pm10:.1f} μg/m³",
        f"{st.session_state.tvoc:.0f} μg/m³",
        f"{st.session_state.formaldehyde:.1f} μg/m³",
        f"{st.session_state.co:.1f} ppm",
        f"{st.session_state.oxygen:.2f} %",
        f"{st.session_state.noise:.1f} dBA",
        f"{st.session_state.light:.0f} lux",
        f"{actions['density']:.1f} m²/person" if people > 0 else "empty"
    ],
    "Coworking target": [
        "< 1000 ppm",
        "20–24 °C",
        "40–60 %",
        "< 15 μg/m³",
        "< 45 μg/m³",
        "< 500 μg/m³",
        "< 50 μg/m³",
        "< 9 ppm",
        "> 19.5 %",
        "< 55 dBA",
        "≈ 500 lux",
        "4–6 m²/person"
    ],
    "Controlled by": [
        "Ventilation",
        "Heating / cooling / blinds",
        "Ventilation / humidification",
        "Filtration / outdoor air intake",
        "Filtration / cleaning",
        "Ventilation / carbon filter",
        "Ventilation / source control",
        "Ventilation / alarm",
        "Ventilation / alarm",
        "Sound masking / zoning",
        "Lighting / blinds",
        "Occupancy management"
    ]
})

st.dataframe(sensor_df, use_container_width=True, hide_index=True)


# -----------------------------
# CHARTS
# -----------------------------

df = pd.DataFrame(st.session_state.history)

if len(df) > 2:
    tab1, tab2, tab3 = st.tabs([
        "Indoor environment",
        "Air pollutants",
        "Energy and actions"
    ])

    with tab1:
        fig_co2 = go.Figure()

        fig_co2.add_trace(go.Scatter(
            x=df["Step"],
            y=df["CO2"],
            mode="lines",
            name="CO₂"
        ))

        fig_co2.add_hline(
            y=config["co2_warn"],
            line_dash="dash",
            annotation_text="CO₂ warning threshold"
        )

        fig_co2.add_hline(
            y=config["co2_crit"],
            line_dash="dot",
            annotation_text="CO₂ critical threshold"
        )

        fig_co2.update_layout(
            height=360,
            xaxis_title="Simulation step",
            yaxis_title="CO₂, ppm"
        )

        st.plotly_chart(fig_co2, use_container_width=True)

        fig_temp = go.Figure()

        fig_temp.add_trace(go.Scatter(
            x=df["Step"],
            y=df["Temperature"],
            mode="lines",
            name="Temperature"
        ))

        fig_temp.add_hline(y=config["temp_min"], line_dash="dash")
        fig_temp.add_hline(y=config["temp_max"], line_dash="dash")

        fig_temp.update_layout(
            height=320,
            xaxis_title="Simulation step",
            yaxis_title="Temperature, °C"
        )

        st.plotly_chart(fig_temp, use_container_width=True)

    with tab2:
        fig_pollution = go.Figure()

        fig_pollution.add_trace(go.Scatter(
            x=df["Step"],
            y=df["PM2.5"],
            mode="lines",
            name="PM2.5"
        ))

        fig_pollution.add_trace(go.Scatter(
            x=df["Step"],
            y=df["TVOC"],
            mode="lines",
            name="TVOC"
        ))

        fig_pollution.add_trace(go.Scatter(
            x=df["Step"],
            y=df["Formaldehyde"],
            mode="lines",
            name="Formaldehyde"
        ))

        fig_pollution.update_layout(
            height=360,
            xaxis_title="Simulation step",
            yaxis_title="Value"
        )

        st.plotly_chart(fig_pollution, use_container_width=True)

    with tab3:
        fig_energy = go.Figure()

        fig_energy.add_trace(go.Scatter(
            x=df["Step"],
            y=df["Energy"],
            mode="lines",
            name="Energy use, kW"
        ))

        fig_energy.add_trace(go.Scatter(
            x=df["Step"],
            y=df["Battery"],
            mode="lines",
            name="Battery, %"
        ))

        fig_energy.update_layout(
            height=360,
            xaxis_title="Simulation step",
            yaxis_title="Value"
        )

        st.plotly_chart(fig_energy, use_container_width=True)

        fig_actions = go.Figure()

        fig_actions.add_trace(go.Scatter(
            x=df["Step"],
            y=df["Ventilation"],
            mode="lines",
            name="Ventilation"
        ))

        fig_actions.add_trace(go.Scatter(
            x=df["Step"],
            y=df["Filtration"],
            mode="lines",
            name="Filtration"
        ))

        fig_actions.add_trace(go.Scatter(
            x=df["Step"],
            y=df["Lighting"],
            mode="lines",
            name="Lighting"
        ))

        fig_actions.update_layout(
            height=320,
            xaxis_title="Simulation step",
            yaxis_title="System action, %"
        )

        st.plotly_chart(fig_actions, use_container_width=True)


# -----------------------------
# AUTO REFRESH
# -----------------------------

if auto_run:
    time.sleep(refresh_speed)
    st.rerun()


import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Coworking + Shelter Digital Twin",
    page_icon="🏫",
    layout="wide"
)


# ============================================================
# DATA LOADING
# ============================================================

DEFAULT_DATA_PATHS = [
    Path("data/coworking_shelter_sensor_dataset.csv"),
    Path("coworking_shelter_sensor_dataset.csv"),
]


@st.cache_data
def load_dataset_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return prepare_dataset(df)


@st.cache_data
def load_dataset_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return prepare_dataset(df)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_columns = [
        "occupancy",
        "active_capacity",
        "coworking_capacity",
        "emergency_capacity",
        "floor_area_m2",
        "co2_ppm",
        "indoor_temperature_c",
        "relative_humidity_percent",
        "pm25_ug_m3",
        "pm10_ug_m3",
        "tvoc_ug_m3",
        "formaldehyde_ug_m3",
        "co_ppm",
        "oxygen_percent",
        "noise_dba",
        "indoor_light_lux",
        "daylight_lux",
        "outside_temperature_c",
        "outdoor_pm25_ug_m3",
        "battery_percent",
        "energy_use_kw",
        "comfort_score",
        "ventilation_level_percent",
        "outside_air_intake_percent",
        "filtration_level_percent",
        "heating_level_percent",
        "cooling_level_percent",
        "lighting_level_percent",
        "blinds_closed_percent",
        "estimated_air_changes_per_hour",
        "estimated_airflow_lps",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    binary_columns = [
        "alarm",
        "grid_available",
        "backup_power_only",
        "smoke_detected",
        "water_leak_detected",
        "door_open",
        "window_open",
        "ventilation_fault",
        "is_unsafe",
    ]

    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Fallbacks for missing columns
    if "mode" not in df.columns:
        df["mode"] = "coworking"

    if "planned_scenario" not in df.columns:
        df["planned_scenario"] = "normal"

    if "active_capacity" not in df.columns:
        if "coworking_capacity" in df.columns:
            df["active_capacity"] = df["coworking_capacity"]
        elif "shelter_capacity" in df.columns:
            df["active_capacity"] = df["shelter_capacity"]
        else:
            df["active_capacity"] = 60

    if "floor_area_m2" not in df.columns:
        df["floor_area_m2"] = 150

    return df.reset_index(drop=True)


def find_default_dataset_path():
    for path in DEFAULT_DATA_PATHS:
        if path.exists():
            return path
    return None


def get_value(row: pd.Series, col: str, default=0):
    if col in row.index and pd.notna(row[col]):
        return row[col]
    return default


# ============================================================
# THRESHOLDS
# ============================================================

def get_thresholds(mode: str) -> dict:
    if mode == "emergency_shelter":
        return {
            "label": "Emergency shelter mode",
            "goal": "Healthy minimum conditions + maximum battery autonomy",
            "co2_warn": 1500,
            "co2_crit": 2200,
            "temp_min": 16,
            "temp_max": 28,
            "humidity_min": 30,
            "humidity_max": 70,
            "pm25_warn": 25,
            "pm25_crit": 75,
            "pm10_warn": 75,
            "tvoc_warn": 700,
            "tvoc_crit": 1200,
            "hcho_warn": 80,
            "co_warn": 9,
            "co_crit": 30,
            "oxygen_min": 19.5,
            "noise_warn": 75,
            "lux_target": 100,
            "battery_warn": 50,
            "battery_crit": 10,
            "base_ventilation": 15,
            "high_ventilation": 55,
            "glare_lux": 1200,
            "outdoor_pm25_high": 25,
        }

    return {
        "label": "Coworking mode",
        "goal": "Comfort + productivity + energy efficiency",
        "co2_warn": 1000,
        "co2_crit": 1500,
        "temp_min": 20,
        "temp_max": 24,
        "humidity_min": 40,
        "humidity_max": 60,
        "pm25_warn": 15,
        "pm25_crit": 50,
        "pm10_warn": 45,
        "tvoc_warn": 500,
        "tvoc_crit": 1000,
        "hcho_warn": 50,
        "co_warn": 9,
        "co_crit": 30,
        "oxygen_min": 19.5,
        "noise_warn": 55,
        "lux_target": 500,
        "battery_warn": 30,
        "battery_crit": 10,
        "base_ventilation": 25,
        "high_ventilation": 75,
        "glare_lux": 1000,
        "outdoor_pm25_high": 25,
    }


# ============================================================
# DIGITAL TWIN DECISION LOGIC
# ============================================================

def decide_actions(row: pd.Series, thresholds: dict) -> dict:
    mode = str(get_value(row, "mode", "coworking"))

    occupancy = float(get_value(row, "occupancy", 0))
    active_capacity = max(float(get_value(row, "active_capacity", 60)), 1)
    floor_area = max(float(get_value(row, "floor_area_m2", 150)), 1)

    occupancy_ratio = occupancy / active_capacity
    area_per_person = floor_area / occupancy if occupancy > 0 else np.inf

    co2 = float(get_value(row, "co2_ppm", 420))
    temp = float(get_value(row, "indoor_temperature_c", 22))
    humidity = float(get_value(row, "relative_humidity_percent", 45))
    pm25 = float(get_value(row, "pm25_ug_m3", 5))
    pm10 = float(get_value(row, "pm10_ug_m3", 10))
    tvoc = float(get_value(row, "tvoc_ug_m3", 250))
    hcho = float(get_value(row, "formaldehyde_ug_m3", 10))
    co = float(get_value(row, "co_ppm", 0))
    oxygen = float(get_value(row, "oxygen_percent", 20.9))
    noise = float(get_value(row, "noise_dba", 40))
    daylight = float(get_value(row, "daylight_lux", 300))
    outside_temp = float(get_value(row, "outside_temperature_c", temp))
    outdoor_pm25 = float(get_value(row, "outdoor_pm25_ug_m3", 5))
    battery = float(get_value(row, "battery_percent", 100))

    smoke = int(get_value(row, "smoke_detected", 0)) == 1
    water_leak = int(get_value(row, "water_leak_detected", 0)) == 1
    backup_power_only = int(get_value(row, "backup_power_only", 0)) == 1
    ventilation_fault = int(get_value(row, "ventilation_fault", 0)) == 1

    ventilation = 5 if occupancy == 0 else thresholds["base_ventilation"]
    outside_air_intake = ventilation
    filtration = 15
    heating = 0
    cooling = 0
    lighting = 0
    blinds = 20
    sound_masking = 0
    emergency_lights = 0
    smoke_exhaust = 0

    system_notes = []

    # CO2 / occupancy
    if co2 > thresholds["co2_warn"]:
        ventilation = max(ventilation, thresholds["high_ventilation"])
        system_notes.append("CO₂ is above the warning threshold → ventilation increased.")

    if co2 > thresholds["co2_crit"]:
        ventilation = 100
        system_notes.append("CO₂ is critical → maximum ventilation required.")

    if occupancy_ratio > 0.85:
        ventilation = max(ventilation, thresholds["high_ventilation"])
        system_notes.append("Occupancy is high → ventilation increased.")

    if occupancy_ratio > 1.0:
        system_notes.append("Space is overcrowded → occupancy warning activated.")

    # Air pollution
    if pm25 > thresholds["pm25_warn"] or pm10 > thresholds["pm10_warn"]:
        filtration = max(filtration, 80)
        system_notes.append("Particulate matter is high → filtration increased.")

    if pm25 > thresholds["pm25_crit"]:
        filtration = 100
        system_notes.append("PM2.5 is critical → maximum filtration required.")

    if tvoc > thresholds["tvoc_warn"] or hcho > thresholds["hcho_warn"]:
        ventilation = max(ventilation, 70)
        filtration = 100
        system_notes.append("VOC/formaldehyde is high → filtration and ventilation increased.")

    if co > thresholds["co_warn"] or oxygen < thresholds["oxygen_min"]:
        ventilation = 100
        filtration = 100
        system_notes.append("CO/O₂ safety risk → maximum air exchange required.")

    # Outdoor pollution logic
    if outdoor_pm25 > thresholds["outdoor_pm25_high"] and co2 < thresholds["co2_crit"]:
        outside_air_intake = min(ventilation, 30)
        filtration = 100
        system_notes.append("Outdoor PM2.5 is high → outside air intake limited, filtration increased.")
    else:
        outside_air_intake = ventilation

    # Temperature
    if temp < thresholds["temp_min"]:
        heating = min(100, (thresholds["temp_min"] - temp) * 30)
        system_notes.append("Indoor temperature is too low → heating activated.")

    if temp > thresholds["temp_max"]:
        cooling = min(100, (temp - thresholds["temp_max"]) * 30)
        ventilation = max(ventilation, 45)
        system_notes.append("Indoor temperature is too high → cooling and ventilation increased.")

    # Emergency mode: less comfort-focused HVAC
    if mode == "emergency_shelter":
        heating *= 0.5
        cooling *= 0.5
        emergency_lights = 60
        system_notes.append("Emergency mode → HVAC is limited to save energy.")

    # Lighting
    target_lux = thresholds["lux_target"]

    if occupancy == 0:
        target_lux = 30

    if mode == "emergency_shelter" and battery < 30:
        target_lux = 50
        system_notes.append("Low battery in emergency mode → light level reduced.")

    artificial_needed = max(0, target_lux - daylight)
    lighting = min(100, artificial_needed / 650 * 100)

    # Blinds
    if daylight > thresholds["glare_lux"]:
        blinds = 80
        system_notes.append("Too much daylight/glare → blinds closed.")

    if temp > thresholds["temp_max"] and outside_temp > temp:
        blinds = 90
        system_notes.append("Outdoor heat is high → blinds closed to reduce solar gains.")

    # Noise
    if mode == "coworking" and noise > thresholds["noise_warn"]:
        sound_masking = 40
        system_notes.append("Noise is high → quiet-zone alert / sound masking activated.")

    # Energy saving
    energy_saving = False
    if backup_power_only or battery < thresholds["battery_warn"] or mode == "emergency_shelter" or occupancy == 0:
        energy_saving = True

    if energy_saving:
        lighting *= 0.7
        sound_masking *= 0.5
        system_notes.append("Energy-saving logic is active.")

    # Safety
    alarm_state = "OFF"
    exits_state = "Normal access"
    outlets_state = "ON"
    maintenance_alert = "OFF"

    if mode == "emergency_shelter":
        exits_state = "Emergency exits unlocked"

    if ventilation_fault:
        maintenance_alert = "Ventilation fault"
        system_notes.append("Ventilation fault detected → maintenance alert.")

    if smoke:
        alarm_state = "FIRE ALARM"
        ventilation = 0
        outside_air_intake = 0
        smoke_exhaust = 100
        filtration = 100
        emergency_lights = 100
        exits_state = "Emergency exits unlocked"
        system_notes.append("Smoke detected → fire safety scenario activated.")

    if water_leak:
        outlets_state = "OFF in affected zone"
        system_notes.append("Water leak detected → affected outlets switched off.")

    return {
        "occupancy_ratio": occupancy_ratio,
        "area_per_person": area_per_person,
        "ventilation": float(np.clip(ventilation, 0, 100)),
        "outside_air_intake": float(np.clip(outside_air_intake, 0, 100)),
        "filtration": float(np.clip(filtration, 0, 100)),
        "heating": float(np.clip(heating, 0, 100)),
        "cooling": float(np.clip(cooling, 0, 100)),
        "lighting": float(np.clip(lighting, 0, 100)),
        "blinds": float(np.clip(blinds, 0, 100)),
        "sound_masking": float(np.clip(sound_masking, 0, 100)),
        "emergency_lights": float(np.clip(emergency_lights, 0, 100)),
        "smoke_exhaust": float(np.clip(smoke_exhaust, 0, 100)),
        "energy_saving": energy_saving,
        "alarm_state": alarm_state,
        "exits_state": exits_state,
        "outlets_state": outlets_state,
        "maintenance_alert": maintenance_alert,
        "notes": system_notes if system_notes else ["All indicators are within the target range."]
    }


def classify_status(row: pd.Series, thresholds: dict, actions: dict):
    co2 = float(get_value(row, "co2_ppm", 420))
    temp = float(get_value(row, "indoor_temperature_c", 22))
    humidity = float(get_value(row, "relative_humidity_percent", 45))
    pm25 = float(get_value(row, "pm25_ug_m3", 5))
    tvoc = float(get_value(row, "tvoc_ug_m3", 250))
    co = float(get_value(row, "co_ppm", 0))
    oxygen = float(get_value(row, "oxygen_percent", 20.9))
    battery = float(get_value(row, "battery_percent", 100))
    noise = float(get_value(row, "noise_dba", 40))

    smoke = int(get_value(row, "smoke_detected", 0)) == 1
    water_leak = int(get_value(row, "water_leak_detected", 0)) == 1

    critical = (
        smoke
        or co2 > thresholds["co2_crit"]
        or pm25 > thresholds["pm25_crit"]
        or tvoc > thresholds["tvoc_crit"]
        or co > thresholds["co_crit"]
        or oxygen < thresholds["oxygen_min"]
        or battery < thresholds["battery_crit"]
    )

    warning = (
        co2 > thresholds["co2_warn"]
        or temp < thresholds["temp_min"]
        or temp > thresholds["temp_max"]
        or humidity < thresholds["humidity_min"]
        or humidity > thresholds["humidity_max"]
        or pm25 > thresholds["pm25_warn"]
        or tvoc > thresholds["tvoc_warn"]
        or noise > thresholds["noise_warn"]
        or water_leak
        or actions["occupancy_ratio"] > 0.85
    )

    if critical:
        return "Critical", "#F44336"
    if warning:
        return "Warning", "#FFC107"
    return "Optimal", "#4CAF50"


# ============================================================
# VISUAL HELPERS
# ============================================================

def pct_bar(label: str, value: float, caption: str = ""):
    value = float(np.clip(value, 0, 100))
    st.write(f"**{label}: {value:.0f}%**")
    st.progress(int(value))
    if caption:
        st.caption(caption)


def status_badge(text: str, color: str):
    st.markdown(
        f"""
        <div style="
            padding: 0.6rem 0.9rem;
            border-radius: 0.7rem;
            background-color: {color}33;
            border: 1px solid {color};
            color: {color};
            font-weight: 700;
            text-align: center;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_room_figure(row: pd.Series, actions: dict, status: str, status_color: str):
    occupancy = int(get_value(row, "occupancy", 0))
    mode = str(get_value(row, "mode", "coworking"))
    scenario = str(get_value(row, "planned_scenario", "normal"))

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=12,
        y1=7,
        line=dict(color="black", width=3),
        fillcolor=status_color,
        opacity=0.45,
    )

    # Zoning blocks
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=4, y1=3, line=dict(color="#555"), fillcolor="rgba(255,255,255,0.35)")
    fig.add_shape(type="rect", x0=4.5, y0=0.5, x1=8, y1=3, line=dict(color="#555"), fillcolor="rgba(255,255,255,0.35)")
    fig.add_shape(type="rect", x0=8.5, y0=0.5, x1=11.5, y1=3, line=dict(color="#555"), fillcolor="rgba(255,255,255,0.35)")

    fig.add_annotation(x=2.25, y=1.75, text="Focus zone", showarrow=False, font=dict(size=11))
    fig.add_annotation(x=6.25, y=1.75, text="Group zone", showarrow=False, font=dict(size=11))
    fig.add_annotation(x=10, y=1.75, text="Shelter zone", showarrow=False, font=dict(size=11))

    # People dots
    dots = min(occupancy, 220)
    if dots > 0:
        rng = np.random.default_rng(42)
        x = rng.uniform(0.8, 11.2, dots)
        y = rng.uniform(3.3, 6.4, dots)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=7, color="black"),
                name="People",
                hovertemplate="Person<extra></extra>",
            )
        )

    # Ventilation block
    vent_color = "#2196F3" if actions["ventilation"] > 0 else "#9E9E9E"
    fig.add_shape(
        type="rect",
        x0=11.45,
        y0=3.3,
        x1=12.4,
        y1=5.6,
        line=dict(color=vent_color, width=3),
        fillcolor=vent_color,
        opacity=0.85,
    )
    fig.add_annotation(x=12.7, y=4.45, text="Vent", showarrow=False, font=dict(size=12))

    # Door
    door_open = int(get_value(row, "door_open", 0))
    door_color = "#4CAF50" if door_open else "#795548"
    fig.add_shape(
        type="rect",
        x0=5.2,
        y0=-0.1,
        x1=6.8,
        y1=0.15,
        line=dict(color=door_color, width=3),
        fillcolor=door_color,
    )
    fig.add_annotation(x=6, y=-0.35, text="Door open" if door_open else "Door closed", showarrow=False, font=dict(size=11))

    title = f"{mode} | {scenario} | {status} | Ventilation {actions['ventilation']:.0f}% | Filtration {actions['filtration']:.0f}%"

    fig.add_annotation(x=6, y=7.45, text=title, showarrow=False, font=dict(size=15))

    fig.update_layout(
        height=460,
        xaxis=dict(visible=False, range=[-0.5, 13.4]),
        yaxis=dict(visible=False, range=[-0.6, 7.8]),
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dataset playback")

default_path = find_default_dataset_path()
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    dataset = load_dataset_from_upload(uploaded_file)
    st.sidebar.success("Uploaded dataset is active.")
elif default_path is not None:
    dataset = load_dataset_from_path(str(default_path))
    st.sidebar.success(f"Loaded: {default_path}")
else:
    st.error(
        "Dataset not found. Put `coworking_shelter_sensor_dataset.csv` into the `data/` folder "
        "or upload it through the sidebar."
    )
    st.stop()

# Filters
available_modes = sorted(dataset["mode"].dropna().unique().tolist())
selected_modes = st.sidebar.multiselect(
    "Modes",
    available_modes,
    default=available_modes,
)

filtered = dataset[dataset["mode"].isin(selected_modes)].copy()

available_scenarios = sorted(filtered["planned_scenario"].dropna().unique().tolist())
selected_scenarios = st.sidebar.multiselect(
    "Scenarios",
    available_scenarios,
    default=available_scenarios,
)

filtered = filtered[filtered["planned_scenario"].isin(selected_scenarios)].reset_index(drop=True)

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

if st.session_state.row_index >= len(filtered):
    st.session_state.row_index = 0

playback_mode = st.sidebar.radio(
    "Playback mode",
    ["Manual", "Auto"],
    horizontal=True,
)

speed = st.sidebar.slider("Refresh speed, seconds", 0.2, 3.0, 0.8)

st.session_state.row_index = st.sidebar.slider(
    "Current dataset row",
    0,
    len(filtered) - 1,
    int(st.session_state.row_index),
)

history_window = st.sidebar.slider("Chart window, rows", 20, 400, 120, step=20)

if st.sidebar.button("Restart playback"):
    st.session_state.row_index = 0
    st.rerun()

row = filtered.iloc[st.session_state.row_index]
mode = str(get_value(row, "mode", "coworking"))
thresholds = get_thresholds(mode)
actions = decide_actions(row, thresholds)
status, status_color = classify_status(row, thresholds, actions)


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title("Adaptive Student Coworking Digital Twin")
st.caption(
    "Dataset rows are replayed as a simulated real-time sensor stream. "
    "The digital twin reads current sensor values and calculates recommended control actions."
)

top1, top2, top3, top4 = st.columns([1.2, 1.2, 1.2, 1.2])

with top1:
    status_badge(f"Status: {status}", status_color)

with top2:
    st.metric("Mode", thresholds["label"])

with top3:
    st.metric("Scenario", str(get_value(row, "planned_scenario", "normal")))

with top4:
    ts = get_value(row, "timestamp", "unknown")
    st.metric("Timestamp", str(ts))

st.info(f"Control goal: {thresholds['goal']}")

# Key metrics
metric_cols = st.columns(8)

with metric_cols[0]:
    st.metric("People", int(get_value(row, "occupancy", 0)))

with metric_cols[1]:
    st.metric("Capacity use", f"{actions['occupancy_ratio'] * 100:.0f}%")

with metric_cols[2]:
    st.metric("CO₂", f"{get_value(row, 'co2_ppm', 0):.0f} ppm")

with metric_cols[3]:
    st.metric("Temperature", f"{get_value(row, 'indoor_temperature_c', 0):.1f} °C")

with metric_cols[4]:
    st.metric("Humidity", f"{get_value(row, 'relative_humidity_percent', 0):.0f}%")

with metric_cols[5]:
    st.metric("PM2.5", f"{get_value(row, 'pm25_ug_m3', 0):.1f} μg/m³")

with metric_cols[6]:
    st.metric("Battery", f"{get_value(row, 'battery_percent', 0):.0f}%")

with metric_cols[7]:
    st.metric("Comfort", f"{get_value(row, 'comfort_score', 0):.0f}/100")


# ============================================================
# ROOM + ACTIONS
# ============================================================

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Room state")
    fig_room = make_room_figure(row, actions, status, status_color)
    st.plotly_chart(fig_room, use_container_width=True)

with right:
    st.subheader("Digital twin recommended actions")

    pct_bar("Ventilation", actions["ventilation"], "Controls CO₂, humidity and heat removal.")
    pct_bar("Outside air intake", actions["outside_air_intake"], "Limited if outdoor PM2.5 is high.")
    pct_bar("Air filtration", actions["filtration"], "Controls PM2.5, PM10, VOC and formaldehyde.")
    pct_bar("Heating", actions["heating"])
    pct_bar("Cooling", actions["cooling"])
    pct_bar("Lighting", actions["lighting"])
    pct_bar("Blinds closed", actions["blinds"])
    pct_bar("Sound masking", actions["sound_masking"])

    if actions["emergency_lights"] > 0:
        pct_bar("Emergency lights", actions["emergency_lights"])

    if actions["smoke_exhaust"] > 0:
        pct_bar("Smoke exhaust", actions["smoke_exhaust"])

    st.write(f"**Energy saving:** {'ON' if actions['energy_saving'] else 'OFF'}")
    st.write(f"**Alarm:** {actions['alarm_state']}")
    st.write(f"**Exits:** {actions['exits_state']}")
    st.write(f"**Power outlets:** {actions['outlets_state']}")
    st.write(f"**Maintenance:** {actions['maintenance_alert']}")


# ============================================================
# SENSOR TABLE + DECISION EXPLANATION
# ============================================================

tab_sensors, tab_explanation, tab_compare = st.tabs(
    ["Sensor values", "Why did the system act?", "Dataset equipment state"]
)

with tab_sensors:
    sensor_table = pd.DataFrame(
        {
            "Indicator": [
                "CO₂",
                "Temperature",
                "Relative humidity",
                "PM2.5",
                "PM10",
                "TVOC",
                "Formaldehyde",
                "Carbon monoxide",
                "Oxygen",
                "Noise",
                "Indoor light",
                "Daylight",
                "Outdoor temperature",
                "Outdoor PM2.5",
                "Area per person",
                "Battery",
                "Energy use",
            ],
            "Current value": [
                f"{get_value(row, 'co2_ppm', 0):.0f} ppm",
                f"{get_value(row, 'indoor_temperature_c', 0):.1f} °C",
                f"{get_value(row, 'relative_humidity_percent', 0):.0f}%",
                f"{get_value(row, 'pm25_ug_m3', 0):.1f} μg/m³",
                f"{get_value(row, 'pm10_ug_m3', 0):.1f} μg/m³",
                f"{get_value(row, 'tvoc_ug_m3', 0):.0f} μg/m³",
                f"{get_value(row, 'formaldehyde_ug_m3', 0):.1f} μg/m³",
                f"{get_value(row, 'co_ppm', 0):.1f} ppm",
                f"{get_value(row, 'oxygen_percent', 20.9):.2f}%",
                f"{get_value(row, 'noise_dba', 0):.1f} dBA",
                f"{get_value(row, 'indoor_light_lux', 0):.0f} lux",
                f"{get_value(row, 'daylight_lux', 0):.0f} lux",
                f"{get_value(row, 'outside_temperature_c', 0):.1f} °C",
                f"{get_value(row, 'outdoor_pm25_ug_m3', 0):.1f} μg/m³",
                "empty" if np.isinf(actions["area_per_person"]) else f"{actions['area_per_person']:.1f} m²/person",
                f"{get_value(row, 'battery_percent', 0):.0f}%",
                f"{get_value(row, 'energy_use_kw', 0):.2f} kW",
            ],
            "Target in current mode": [
                f"< {thresholds['co2_warn']} ppm; critical > {thresholds['co2_crit']} ppm",
                f"{thresholds['temp_min']}–{thresholds['temp_max']} °C",
                f"{thresholds['humidity_min']}–{thresholds['humidity_max']}%",
                f"< {thresholds['pm25_warn']} μg/m³",
                f"< {thresholds['pm10_warn']} μg/m³",
                f"< {thresholds['tvoc_warn']} μg/m³",
                f"< {thresholds['hcho_warn']} μg/m³",
                f"< {thresholds['co_warn']} ppm",
                f"> {thresholds['oxygen_min']}%",
                f"< {thresholds['noise_warn']} dBA",
                f"≈ {thresholds['lux_target']} lux",
                "Used for daylight harvesting",
                "Used for heating/cooling decision",
                f"< {thresholds['outdoor_pm25_high']} μg/m³ preferred for outside intake",
                "Coworking: 4–6 m²/person; shelter can be denser",
                f"Warning < {thresholds['battery_warn']}%",
                "Minimize without harming comfort/health",
            ],
        }
    )

    st.dataframe(sensor_table, use_container_width=True, hide_index=True)

with tab_explanation:
    st.write("**Decision logic explanation:**")
    for note in actions["notes"]:
        st.write(f"- {note}")

    trigger = get_value(row, "primary_trigger", "normal")
    triggers = get_value(row, "triggers", "normal")
    st.write("**Dataset labels:**")
    st.write(f"- Primary trigger: `{trigger}`")
    st.write(f"- Triggers: `{triggers}`")
    st.write(f"- Comfort level: `{get_value(row, 'comfort_level', 'unknown')}`")
    st.write(f"- Unsafe label: `{int(get_value(row, 'is_unsafe', 0))}`")

with tab_compare:
    comparison_rows = [
        ("Ventilation", "ventilation_level_percent", actions["ventilation"]),
        ("Outside air intake", "outside_air_intake_percent", actions["outside_air_intake"]),
        ("Filtration", "filtration_level_percent", actions["filtration"]),
        ("Heating", "heating_level_percent", actions["heating"]),
        ("Cooling", "cooling_level_percent", actions["cooling"]),
        ("Lighting", "lighting_level_percent", actions["lighting"]),
        ("Blinds closed", "blinds_closed_percent", actions["blinds"]),
    ]

    comparison = []
    for label, col, recommended in comparison_rows:
        dataset_value = get_value(row, col, np.nan)
        comparison.append(
            {
                "System": label,
                "Dataset equipment state": "missing" if pd.isna(dataset_value) else f"{dataset_value:.0f}%",
                "Digital twin recommendation": f"{recommended:.0f}%",
            }
        )

    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    st.caption(
        "Dataset equipment state can be interpreted as the recorded/current state of equipment. "
        "Digital twin recommendation is calculated live from sensor values."
    )


# ============================================================
# CHARTS
# ============================================================

st.subheader("Time-series playback")

end = st.session_state.row_index + 1
start = max(0, end - history_window)
chart_df = filtered.iloc[start:end].copy()

if "timestamp" in chart_df.columns and chart_df["timestamp"].notna().any():
    x_axis = chart_df["timestamp"]
    x_title = "Time"
else:
    x_axis = chart_df.index
    x_title = "Dataset row"

tab_env, tab_air, tab_energy, tab_actions = st.tabs(
    ["Environment", "Air quality", "Energy", "Actions"]
)

with tab_env:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_axis, y=chart_df["co2_ppm"], mode="lines", name="CO₂, ppm"))
    fig.add_hline(y=thresholds["co2_warn"], line_dash="dash", annotation_text="CO₂ warning")
    fig.add_hline(y=thresholds["co2_crit"], line_dash="dot", annotation_text="CO₂ critical")

    fig.update_layout(height=350, xaxis_title=x_title, yaxis_title="CO₂, ppm")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_axis, y=chart_df["indoor_temperature_c"], mode="lines", name="Temperature, °C"))
    fig2.add_hline(y=thresholds["temp_min"], line_dash="dash", annotation_text="Min target")
    fig2.add_hline(y=thresholds["temp_max"], line_dash="dash", annotation_text="Max target")

    fig2.update_layout(height=330, xaxis_title=x_title, yaxis_title="Temperature, °C")
    st.plotly_chart(fig2, use_container_width=True)

with tab_air:
    fig = go.Figure()

    for col, name in [
        ("pm25_ug_m3", "PM2.5"),
        ("pm10_ug_m3", "PM10"),
        ("tvoc_ug_m3", "TVOC"),
        ("formaldehyde_ug_m3", "Formaldehyde"),
    ]:
        if col in chart_df.columns:
            fig.add_trace(go.Scatter(x=x_axis, y=chart_df[col], mode="lines", name=name))

    fig.update_layout(height=380, xaxis_title=x_title, yaxis_title="Pollutant value")
    st.plotly_chart(fig, use_container_width=True)

with tab_energy:
    fig = go.Figure()

    if "battery_percent" in chart_df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=chart_df["battery_percent"], mode="lines", name="Battery, %"))

    if "energy_use_kw" in chart_df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=chart_df["energy_use_kw"], mode="lines", name="Energy use, kW"))

    fig.update_layout(height=360, xaxis_title=x_title, yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

with tab_actions:
    fig = go.Figure()

    action_columns = [
        ("ventilation_level_percent", "Dataset ventilation"),
        ("filtration_level_percent", "Dataset filtration"),
        ("lighting_level_percent", "Dataset lighting"),
        ("cooling_level_percent", "Dataset cooling"),
        ("heating_level_percent", "Dataset heating"),
    ]

    for col, name in action_columns:
        if col in chart_df.columns:
            fig.add_trace(go.Scatter(x=x_axis, y=chart_df[col], mode="lines", name=name))

    fig.update_layout(height=380, xaxis_title=x_title, yaxis_title="Equipment state, %")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RAW ROW
# ============================================================

with st.expander("Show current dataset row"):
    st.dataframe(pd.DataFrame(row).rename(columns={row.name: "value"}), use_container_width=True)


# ============================================================
# AUTO PLAYBACK
# ============================================================

if playback_mode == "Auto":
    time.sleep(speed)
    st.session_state.row_index = (st.session_state.row_index + 1) % len(filtered)
    st.rerun()

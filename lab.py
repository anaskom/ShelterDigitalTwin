
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Adaptive Coworking + Shelter Digital Twin",
    page_icon="🏫",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_DATA_PATHS = [
    Path("data/coworking_shelter_sensor_dataset.csv"),
    Path("coworking_shelter_sensor_dataset.csv"),
]

HORIZON_STEPS = 2          # 2 steps * 5 min = 10 min forecast
STEP_MINUTES = 5


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(value, low, high):
    return float(np.clip(value, low, high))


def status_color_from_state(state, thresholds):
    if (
        state["co2_ppm"] > thresholds["co2_crit"]
        or state["tvoc_ug_m3"] > thresholds["tvoc_crit"]
        or state["co_ppm"] > thresholds["co_crit"]
        or state["oxygen_percent"] < thresholds["oxygen_min"]
        or state["battery_percent"] < thresholds["battery_crit"]
        or state.get("smoke_detected", False)
    ):
        return "Critical", "#F44336"

    if (
        state["co2_ppm"] > thresholds["co2_warn"]
        or state["indoor_temperature_c"] < thresholds["temp_min"]
        or state["indoor_temperature_c"] > thresholds["temp_max"]
        or state["relative_humidity_percent"] < thresholds["humidity_min"]
        or state["relative_humidity_percent"] > thresholds["humidity_max"]
        or state["tvoc_ug_m3"] > thresholds["tvoc_warn"]
        or state["noise_dba"] > thresholds["noise_warn"]
        or state["occupancy"] / max(state["active_capacity"], 1) > 0.85
        or state.get("water_leak_detected", False)
    ):
        return "Warning", "#FFC107"

    return "Optimal", "#4CAF50"


def compute_comfort_score(state, thresholds):
    score = 100.0

    # CO2 penalty
    if state["co2_ppm"] > thresholds["co2_warn"]:
        score -= min(30, (state["co2_ppm"] - thresholds["co2_warn"]) / 35)

    # Temperature penalty
    if state["indoor_temperature_c"] < thresholds["temp_min"]:
        score -= min(25, (thresholds["temp_min"] - state["indoor_temperature_c"]) * 6)
    if state["indoor_temperature_c"] > thresholds["temp_max"]:
        score -= min(25, (state["indoor_temperature_c"] - thresholds["temp_max"]) * 6)

    # Humidity penalty
    if state["relative_humidity_percent"] < thresholds["humidity_min"]:
        score -= min(15, (thresholds["humidity_min"] - state["relative_humidity_percent"]) * 0.7)
    if state["relative_humidity_percent"] > thresholds["humidity_max"]:
        score -= min(15, (state["relative_humidity_percent"] - thresholds["humidity_max"]) * 0.7)

    # Pollution / safety / crowding
    if state["pm10_ug_m3"] > thresholds["pm10_warn"]:
        score -= min(15, (state["pm10_ug_m3"] - thresholds["pm10_warn"]) * 0.25)
    if state["tvoc_ug_m3"] > thresholds["tvoc_warn"]:
        score -= min(20, (state["tvoc_ug_m3"] - thresholds["tvoc_warn"]) * 0.025)

    occupancy_ratio = state["occupancy"] / max(state["active_capacity"], 1)
    if occupancy_ratio > 0.85:
        score -= min(15, (occupancy_ratio - 0.85) * 60)

    if state["battery_percent"] < thresholds["battery_warn"]:
        score -= min(10, (thresholds["battery_warn"] - state["battery_percent"]) * 0.4)

    return clamp(score, 0, 100)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dataset_if_available():
    for path in DEFAULT_DATA_PATHS:
        if path.exists():
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df
    return pd.DataFrame()


# ============================================================
# MODES AND THRESHOLDS
# ============================================================

def get_thresholds(space_mode):
    if space_mode == "shelter":
        return {
            "co2_warn": 1500,
            "co2_crit": 2200,
            "temp_min": 16,
            "temp_max": 28,
            "humidity_min": 30,
            "humidity_max": 70,
            "pm10_warn": 75,
            "tvoc_warn": 700,
            "tvoc_crit": 1200,
            "hcho_warn": 80,
            "co_warn": 9,
            "co_crit": 30,
            "oxygen_min": 19.5,
            "noise_warn": 75,
            "battery_warn": 50,
            "battery_crit": 10,
            "base_ventilation": 18,
            "high_ventilation": 60,
            "lux_target": 120,
        }

    return {
        "co2_warn": 1000,
        "co2_crit": 1500,
        "temp_min": 20,
        "temp_max": 24,
        "humidity_min": 40,
        "humidity_max": 60,
        "pm10_warn": 45,
        "tvoc_warn": 500,
        "tvoc_crit": 1000,
        "hcho_warn": 50,
        "co_warn": 9,
        "co_crit": 30,
        "oxygen_min": 19.5,
        "noise_warn": 55,
        "battery_warn": 30,
        "battery_crit": 10,
        "base_ventilation": 25,
        "high_ventilation": 75,
        "lux_target": 500,
    }


def scenario_inputs(simulation_mode, step):
    """
    External world / sensor-driving events.

    This is what happens TO the room:
    - people arrive or leave;
    - outside temperature changes;
    - daylight changes;
    - emergency starts;
    - power outage or ventilation fault happens.

    The indoor state itself is NOT overwritten here.
    It evolves later through the physical response function.
    """
    phase = step % 360
    hour = (8 + step / 12) % 24
    daylight = max(0, 850 * np.sin(np.pi * (hour - 6) / 14))
    outside_temp = 18 + 6 * np.sin(2 * np.pi * (step % 288) / 288)

    base = {
        "space_mode": "coworking",
        "situation": "normal coworking",
        "occupancy": 18,
        "active_capacity": 60,
        "outside_temperature_c": outside_temp,
        "outside_humidity_percent": 50,
        "daylight_lux": daylight,
        "grid_available": True,
        "backup_power_only": False,
        "ventilation_fault": False,
        "smoke_detected": False,
        "water_leak_detected": False,
        "event_pm10": 0,
        "event_tvoc": 0,
        "event_heat": 0,
        "event_noise": 0,
    }

    if simulation_mode == "Coworking only":
        local = step % 240
        base["space_mode"] = "coworking"
        base["active_capacity"] = 60

        if local < 60:
            base["situation"] = "normal coworking"
            base["occupancy"] = int(12 + local * 0.35)
        elif local < 105:
            base["situation"] = "crowded coworking"
            base["occupancy"] = int(42 + 10 * np.sin(local / 8))
            base["event_noise"] = 8
        elif local < 150:
            base["situation"] = "overheating risk"
            base["occupancy"] = 48
            base["outside_temperature_c"] = 31
            base["event_heat"] = 0.35
        elif local < 185:
            base["situation"] = "poor air quality"
            base["occupancy"] = 45
            base["event_tvoc"] = 45
            base["event_pm10"] = 3
        else:
            base["situation"] = "stabilized coworking"
            base["occupancy"] = 22

    elif simulation_mode == "Shelter only":
        local = step % 240
        base["space_mode"] = "shelter"
        base["active_capacity"] = 220
        base["daylight_lux"] = min(base["daylight_lux"], 120)

        if local < 60:
            base["situation"] = "air alarm shelter mode"
            base["occupancy"] = int(90 + local * 1.2)
        elif local < 120:
            base["situation"] = "crowded shelter"
            base["occupancy"] = 175
            base["event_noise"] = 12
        elif local < 170:
            base["situation"] = "backup power mode"
            base["occupancy"] = 160
            base["grid_available"] = False
            base["backup_power_only"] = True
        elif local < 205:
            base["situation"] = "limited ventilation"
            base["occupancy"] = 150
            base["grid_available"] = False
            base["backup_power_only"] = True
            base["ventilation_fault"] = True
        else:
            base["situation"] = "shelter stabilization"
            base["occupancy"] = 115
            base["grid_available"] = True
            base["backup_power_only"] = False

    else:
        # Full transition: coworking -> emergency -> shelter stabilization
        if phase < 75:
            base["space_mode"] = "coworking"
            base["situation"] = "normal coworking"
            base["occupancy"] = int(10 + phase * 0.4)
        elif phase < 125:
            base["space_mode"] = "coworking"
            base["situation"] = "crowded coworking"
            base["occupancy"] = 48
            base["event_noise"] = 8
        elif phase < 165:
            base["space_mode"] = "coworking"
            base["situation"] = "overheating before emergency"
            base["occupancy"] = 45
            base["outside_temperature_c"] = 31
            base["event_heat"] = 0.35
        elif phase < 205:
            base["space_mode"] = "shelter"
            base["situation"] = "emergency transition"
            base["active_capacity"] = 220
            base["occupancy"] = int(80 + (phase - 165) * 2.2)
            base["daylight_lux"] = min(base["daylight_lux"], 100)
            base["event_noise"] = 15
        elif phase < 265:
            base["space_mode"] = "shelter"
            base["situation"] = "power outage in shelter"
            base["active_capacity"] = 220
            base["occupancy"] = 170
            base["grid_available"] = False
            base["backup_power_only"] = True
            base["daylight_lux"] = 60
            base["event_noise"] = 12
        elif phase < 315:
            base["space_mode"] = "shelter"
            base["situation"] = "ventilation fault in shelter"
            base["active_capacity"] = 220
            base["occupancy"] = 150
            base["grid_available"] = False
            base["backup_power_only"] = True
            base["ventilation_fault"] = True
            base["daylight_lux"] = 40
        else:
            base["space_mode"] = "shelter"
            base["situation"] = "shelter stabilization"
            base["active_capacity"] = 220
            base["occupancy"] = 115
            base["grid_available"] = True
            base["backup_power_only"] = False
            base["daylight_lux"] = 80

    return base


# ============================================================
# INITIAL STATE
# ============================================================

def initial_state(simulation_mode):
    if simulation_mode == "Shelter only":
        mode = "shelter"
        capacity = 220
        people = 90
        temp = 23.0
        co2 = 760
        daylight = 80
    else:
        mode = "coworking"
        capacity = 60
        people = 15
        temp = 22.2
        co2 = 620
        daylight = 500

    thresholds = get_thresholds(mode)

    state = {
        "step": 0,
        "space_mode": mode,
        "situation": "initial state",
        "occupancy": people,
        "active_capacity": capacity,
        "floor_area_m2": 150,
        "co2_ppm": co2,
        "indoor_temperature_c": temp,
        "relative_humidity_percent": 45,
        "pm10_ug_m3": 14,
        "tvoc_ug_m3": 240,
        "formaldehyde_ug_m3": 14,
        "co_ppm": 0.4,
        "oxygen_percent": 20.9,
        "noise_dba": 42,
        "indoor_light_lux": 450,
        "daylight_lux": daylight,
        "outside_temperature_c": 18,
        "battery_percent": 100,
        "energy_use_kw": 1.2,
        "comfort_score": 92,
        "grid_available": True,
        "backup_power_only": False,
        "ventilation_fault": False,
        "smoke_detected": False,
        "water_leak_detected": False,
    }

    state["comfort_score"] = compute_comfort_score(state, thresholds)
    return state


# ============================================================
# PREDICTION AND CONTROL
# ============================================================

def baseline_actions(space_mode):
    thresholds = get_thresholds(space_mode)
    return {
        "ventilation": thresholds["base_ventilation"],
        "outside_air_intake": thresholds["base_ventilation"],
        "filtration": 15,
        "heating": 0,
        "cooling": 0,
        "lighting": 20 if space_mode == "coworking" else 10,
        "emergency_lights": 0,
        "smoke_exhaust": 0,
        "energy_saving": space_mode == "shelter",
        "alarm_state": "OFF",
        "exits_state": "Normal access",
        "notes": [],
    }


def decide_actions(state, inputs, forecast_no_action):
    space_mode = inputs["space_mode"]
    thresholds = get_thresholds(space_mode)

    actions = baseline_actions(space_mode)
    notes = []

    co2_now = state["co2_ppm"]
    co2_future = forecast_no_action["co2_ppm"]
    temp_now = state["indoor_temperature_c"]
    temp_future = forecast_no_action["indoor_temperature_c"]
    occupancy_ratio = inputs["occupancy"] / max(inputs["active_capacity"], 1)

    # CO2 proactive control
    if co2_now > thresholds["co2_warn"] or co2_future > thresholds["co2_warn"]:
        actions["ventilation"] = max(actions["ventilation"], thresholds["high_ventilation"])
        notes.append("Prediction shows CO₂ risk → ventilation increased before the state becomes unsafe.")

    if co2_now > thresholds["co2_crit"] or co2_future > thresholds["co2_crit"]:
        actions["ventilation"] = 100
        notes.append("Critical CO₂ risk → maximum ventilation selected.")

    # Occupancy
    if occupancy_ratio > 0.85:
        actions["ventilation"] = max(actions["ventilation"], thresholds["high_ventilation"])
        notes.append("High occupancy → ventilation increased.")

    # Temperature proactive control
    if temp_now > thresholds["temp_max"] or temp_future > thresholds["temp_max"]:
        actions["cooling"] = max(actions["cooling"], min(100, (max(temp_now, temp_future) - thresholds["temp_max"]) * 35))
        actions["ventilation"] = max(actions["ventilation"], 45)
        notes.append("Prediction shows overheating → cooling activated proactively.")

    if temp_now < thresholds["temp_min"] or temp_future < thresholds["temp_min"]:
        actions["heating"] = max(actions["heating"], min(100, (thresholds["temp_min"] - min(temp_now, temp_future)) * 35))
        notes.append("Prediction shows low temperature → heating activated proactively.")

    # Air pollution
    if state["pm10_ug_m3"] > thresholds["pm10_warn"]:
        actions["filtration"] = max(actions["filtration"], 80)
        notes.append("PM10 is high → filtration increased.")

    if state["tvoc_ug_m3"] > thresholds["tvoc_warn"] or state["formaldehyde_ug_m3"] > thresholds["hcho_warn"]:
        actions["filtration"] = 100
        actions["ventilation"] = max(actions["ventilation"], 70)
        notes.append("VOC/formaldehyde is high → filtration and ventilation increased.")

    if state["co_ppm"] > thresholds["co_warn"] or state["oxygen_percent"] < thresholds["oxygen_min"]:
        actions["ventilation"] = 100
        actions["filtration"] = 100
        notes.append("CO/O₂ safety risk → maximum air exchange required.")

    # Lighting
    target_lux = thresholds["lux_target"]
    if inputs["occupancy"] == 0:
        target_lux = 30
    if space_mode == "shelter" and inputs["backup_power_only"]:
        target_lux = 70

    artificial_needed = max(0, target_lux - inputs["daylight_lux"])
    actions["lighting"] = clamp(artificial_needed / 650 * 100, 0, 100)

    # Emergency / energy logic
    if space_mode == "shelter":
        actions["emergency_lights"] = 60
        actions["exits_state"] = "Emergency exits unlocked"
        actions["energy_saving"] = True
        actions["heating"] *= 0.5
        actions["cooling"] *= 0.5
        notes.append("Shelter mode → comfort is reduced, safety and battery autonomy are prioritized.")

    if inputs["backup_power_only"] or state["battery_percent"] < thresholds["battery_warn"]:
        actions["energy_saving"] = True
        actions["lighting"] *= 0.7
        notes.append("Backup power / low battery → energy-saving logic is active.")

    if inputs["ventilation_fault"]:
        actions["ventilation"] = min(actions["ventilation"], 25)
        notes.append("Ventilation fault → ventilation capacity is limited.")

    if inputs["smoke_detected"]:
        actions["alarm_state"] = "FIRE ALARM"
        actions["ventilation"] = 0
        actions["outside_air_intake"] = 0
        actions["smoke_exhaust"] = 100
        actions["filtration"] = 100
        actions["emergency_lights"] = 100
        actions["exits_state"] = "Emergency exits unlocked"
        notes.append("Smoke detected → smoke exhaust and emergency lighting activated.")
    else:
        actions["outside_air_intake"] = actions["ventilation"]

    for key in ["ventilation", "outside_air_intake", "filtration", "heating", "cooling", "lighting", "emergency_lights", "smoke_exhaust"]:
        actions[key] = clamp(actions[key], 0, 100)

    actions["notes"] = notes if notes else ["All indicators are within the target range."]
    return actions


def simulate_physical_response(state, inputs, actions):
    """
    This is the feedback-loop part:
    actions are applied to the room, and the next indoor state is calculated.
    """
    next_state = state.copy()

    # External inputs become measured metadata
    for key, value in inputs.items():
        if key in [
            "space_mode",
            "situation",
            "occupancy",
            "active_capacity",
            "outside_temperature_c",
            "daylight_lux",
            "grid_available",
            "backup_power_only",
            "ventilation_fault",
            "smoke_detected",
            "water_leak_detected",
        ]:
            next_state[key] = value

    people = inputs["occupancy"]
    ventilation = actions["ventilation"]
    filtration = actions["filtration"]
    cooling = actions["cooling"]
    heating = actions["heating"]

    # CO2 dynamics
    co2_generation = people * 1.25
    co2_removal = (ventilation / 100) * (state["co2_ppm"] - 420) * 0.28
    next_state["co2_ppm"] = clamp(state["co2_ppm"] + co2_generation - co2_removal, 420, 5000)

    # Temperature dynamics
    people_heat = people * 0.006
    outside_exchange = (ventilation / 100) * (inputs["outside_temperature_c"] - state["indoor_temperature_c"]) * 0.05
    hvac_effect = (heating / 100) * 1.0 - (cooling / 100) * 1.2
    next_state["indoor_temperature_c"] = clamp(
        state["indoor_temperature_c"] + people_heat + outside_exchange + hvac_effect + inputs["event_heat"],
        10,
        35,
    )

    # Humidity dynamics
    humidity_generation = people * 0.015
    humidity_exchange = (ventilation / 100) * (inputs["outside_humidity_percent"] - state["relative_humidity_percent"]) * 0.04
    next_state["relative_humidity_percent"] = clamp(
        state["relative_humidity_percent"] + humidity_generation + humidity_exchange,
        20,
        90,
    )

    # Pollution dynamics
    next_state["pm10_ug_m3"] = clamp(
        state["pm10_ug_m3"] + people * 0.02 + inputs["event_pm10"] - (filtration / 100) * state["pm10_ug_m3"] * 0.18,
        2,
        200,
    )

    next_state["tvoc_ug_m3"] = clamp(
        state["tvoc_ug_m3"] + people * 0.45 + inputs["event_tvoc"] - (filtration / 100) * state["tvoc_ug_m3"] * 0.12 - (ventilation / 100) * state["tvoc_ug_m3"] * 0.04,
        50,
        2500,
    )

    next_state["formaldehyde_ug_m3"] = clamp(
        state["formaldehyde_ug_m3"] + people * 0.01 - (filtration / 100) * state["formaldehyde_ug_m3"] * 0.05,
        3,
        200,
    )

    # CO and oxygen
    next_state["co_ppm"] = clamp(state["co_ppm"] * (1 - ventilation / 100 * 0.18), 0, 50)
    oxygen_drop = people * 0.0008
    oxygen_restore = (ventilation / 100) * (20.9 - state["oxygen_percent"]) * 0.35
    next_state["oxygen_percent"] = clamp(state["oxygen_percent"] - oxygen_drop + oxygen_restore, 18.5, 21.0)

    # Noise and light
    base_noise = 35 + people * 0.12 + inputs["event_noise"]
    next_state["noise_dba"] = clamp(base_noise, 30, 90)
    next_state["indoor_light_lux"] = clamp(inputs["daylight_lux"] + actions["lighting"] * 6.5, 0, 1200)

    # Energy and battery
    energy_use = (
        0.6
        + actions["ventilation"] * 0.015
        + actions["filtration"] * 0.010
        + actions["heating"] * 0.025
        + actions["cooling"] * 0.030
        + actions["lighting"] * 0.006
        + actions["emergency_lights"] * 0.004
    )
    next_state["energy_use_kw"] = round(energy_use, 2)

    if inputs["grid_available"]:
        next_state["battery_percent"] = clamp(state["battery_percent"] + 0.25, 0, 100)
    else:
        drain = 0.10 + energy_use * 0.18
        next_state["battery_percent"] = clamp(state["battery_percent"] - drain, 0, 100)

    thresholds = get_thresholds(inputs["space_mode"])
    next_state["comfort_score"] = compute_comfort_score(next_state, thresholds)

    return next_state


def forecast_state(state, inputs, actions, steps):
    predicted = state.copy()
    for _ in range(steps):
        predicted = simulate_physical_response(predicted, inputs, actions)
    return predicted


# ============================================================
# SESSION STATE
# ============================================================

def reset_simulation(simulation_mode):
    st.session_state.state = initial_state(simulation_mode)
    st.session_state.history = []
    st.session_state.last_simulation_mode = simulation_mode


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Simulation controls")

simulation_mode = st.sidebar.radio(
    "Simulation mode",
    ["Coworking only", "Shelter only", "Full transition"],
    index=2,
)

speed = st.sidebar.slider("State update speed, seconds", 0.5, 3.0, 1.0)
auto_run = st.sidebar.toggle("Run simulation", value=True)

if "state" not in st.session_state or st.session_state.get("last_simulation_mode") != simulation_mode:
    reset_simulation(simulation_mode)

if st.sidebar.button("Restart simulation"):
    reset_simulation(simulation_mode)
    st.rerun()

if st.sidebar.button("Next state"):
    st.session_state.manual_next = True



# ============================================================
# LOOP CALCULATION HELPER
# ============================================================

def calculate_loop_outputs(current_state, simulation_mode):
    state = current_state.copy()
    step = int(state["step"])
    inputs = scenario_inputs(simulation_mode, step)

    # Current measured metadata
    state["space_mode"] = inputs["space_mode"]
    state["situation"] = inputs["situation"]
    state["occupancy"] = inputs["occupancy"]
    state["active_capacity"] = inputs["active_capacity"]
    state["outside_temperature_c"] = inputs["outside_temperature_c"]
    state["daylight_lux"] = inputs["daylight_lux"]
    state["grid_available"] = inputs["grid_available"]
    state["backup_power_only"] = inputs["backup_power_only"]
    state["ventilation_fault"] = inputs["ventilation_fault"]
    state["smoke_detected"] = inputs["smoke_detected"]
    state["water_leak_detected"] = inputs["water_leak_detected"]

    thresholds = get_thresholds(inputs["space_mode"])
    status, status_color = status_color_from_state(state, thresholds)

    # 1) Prediction without new intervention
    no_action = baseline_actions(inputs["space_mode"])
    forecast_no_action = forecast_state(state, inputs, no_action, HORIZON_STEPS)

    # 2) Controller decides actions using current state + forecast
    actions = decide_actions(state, inputs, forecast_no_action)

    # 3) Predicted result after digital twin action
    forecast_after_action = forecast_state(state, inputs, actions, HORIZON_STEPS)

    # 4) Next physical state after one real step
    next_state = simulate_physical_response(state, inputs, actions)
    next_state["step"] = step + 1



    return {
        "state": state,
        "step": step,
        "inputs": inputs,
        "thresholds": thresholds,
        "status": status,
        "status_color": status_color,
        "forecast_no_action": forecast_no_action,
        "actions": actions,
        "forecast_after_action": forecast_after_action,
        "next_state": next_state,
    }


if st.session_state.pop("manual_next", False):
    manual_outputs = calculate_loop_outputs(st.session_state.state, simulation_mode)
    st.session_state.state = manual_outputs["next_state"]


# ============================================================
# FRAGMENTED UI
# ============================================================

run_every_value = f"{speed}s" if auto_run else None

if hasattr(st, "fragment"):
    def simulation_fragment(func):
        return st.fragment(run_every=run_every_value)(func)
else:
    def simulation_fragment(func):
        return func


@simulation_fragment
def render_simulation_frame():
    outputs = calculate_loop_outputs(st.session_state.state, simulation_mode)

    state = outputs["state"]
    step = outputs["step"]
    inputs = outputs["inputs"]
    thresholds = outputs["thresholds"]
    status = outputs["status"]
    status_color = outputs["status_color"]
    forecast_no_action = outputs["forecast_no_action"]
    actions = outputs["actions"]
    forecast_after_action = outputs["forecast_after_action"]
    next_state = outputs["next_state"]

    # ============================================================
    # MAIN DASHBOARD
    # ============================================================

    st.title("Adaptive Student Coworking Digital Twin")

    mode_label = "Shelter" if state["space_mode"] == "shelter" else "Coworking"

    st.markdown(
        f"""
        <div style="
            padding: 1rem 1.2rem;
            border-radius: 0.8rem;
            border: 1px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.04);
            margin-bottom: 1rem;">
            <div style="font-size: 0.9rem; opacity: 0.75;">Current simulation step</div>
            <div style="font-size: 1.6rem; font-weight: 800;">{state['situation']}</div>
            <div style="font-size: 1rem; opacity: 0.85;">
                Space mode: <b>{mode_label}</b> · Step: <b>{step}</b> · 
                The loop is: sensors → forecast → control action → changed room state → next sensor state.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # STEP 1 — CURRENT SENSOR INPUT
    # ============================================================

    st.subheader("1. Sensor input: current measured room state")

    sensor_cols = st.columns(6)

    with sensor_cols[0]:
        st.metric("People", int(state["occupancy"]))

    with sensor_cols[1]:
        st.metric("Capacity use", f"{state['occupancy'] / max(state['active_capacity'], 1) * 100:.0f}%")

    with sensor_cols[2]:
        st.metric("CO₂", f"{state['co2_ppm']:.0f} ppm")

    with sensor_cols[3]:
        st.metric("Temperature", f"{state['indoor_temperature_c']:.1f} °C")

    with sensor_cols[4]:
        st.metric("Humidity", f"{state['relative_humidity_percent']:.0f}%")

    with sensor_cols[5]:
        st.metric("Comfort", f"{state['comfort_score']:.0f}/100")


    # ============================================================
    # STEPS 2–4 — CLOSED LOOP TABLE
    # ============================================================

    st.subheader("2–4. Forecast, decision and simulated result")

    closed_loop_table = pd.DataFrame(
        {
            "Indicator": ["CO₂", "Temperature", "Humidity", "TVOC", "Battery", "Comfort"],
            "1. Current sensor state": [
                f"{state['co2_ppm']:.0f} ppm",
                f"{state['indoor_temperature_c']:.1f} °C",
                f"{state['relative_humidity_percent']:.0f}%",
                f"{state['tvoc_ug_m3']:.0f} μg/m³",
                f"{state['battery_percent']:.0f}%",
                f"{state['comfort_score']:.0f}/100",
            ],
            "2. Predicted in 10 min without action": [
                f"{forecast_no_action['co2_ppm']:.0f} ppm",
                f"{forecast_no_action['indoor_temperature_c']:.1f} °C",
                f"{forecast_no_action['relative_humidity_percent']:.0f}%",
                f"{forecast_no_action['tvoc_ug_m3']:.0f} μg/m³",
                f"{forecast_no_action['battery_percent']:.0f}%",
                f"{forecast_no_action['comfort_score']:.0f}/100",
            ],
            "4. Predicted in 10 min after action": [
                f"{forecast_after_action['co2_ppm']:.0f} ppm",
                f"{forecast_after_action['indoor_temperature_c']:.1f} °C",
                f"{forecast_after_action['relative_humidity_percent']:.0f}%",
                f"{forecast_after_action['tvoc_ug_m3']:.0f} μg/m³",
                f"{forecast_after_action['battery_percent']:.0f}%",
                f"{forecast_after_action['comfort_score']:.0f}/100",
            ],
        }
    )

    st.dataframe(closed_loop_table, width="stretch", hide_index=True)

    st.caption(
        "Column 2 shows what the system expects if it does nothing. "
        "Column 4 shows the expected state after the digital twin applies the selected control actions."
    )


    # ============================================================
    # ROOM VISUALIZATION
    # ============================================================

    def make_room_figure(state, actions, status_color):
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

        dots = min(int(state["occupancy"]), 220)
        if dots > 0:
            rng = np.random.default_rng(42)
            x = rng.uniform(0.8, 11.2, dots)
            y = rng.uniform(0.8, 6.2, dots)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=7, color="black"),
                    hovertemplate="Person<extra></extra>",
                )
            )

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

        # Main door
        fig.add_shape(
            type="rect",
            x0=5.2,
            y0=-0.1,
            x1=6.8,
            y1=0.15,
            line=dict(color="#795548", width=3),
            fillcolor="#795548",
        )
        fig.add_annotation(x=6, y=-0.35, text="Main door", showarrow=False, font=dict(size=11))

        # Emergency exit
        fig.add_shape(
            type="rect",
            x0=-0.1,
            y0=4.8,
            x1=0.15,
            y1=6.2,
            line=dict(color="#D32F2F", width=3),
            fillcolor="#D32F2F",
        )
        fig.add_annotation(
            x=-0.45,
            y=5.5,
            text="Emergency exit",
            textangle=-90,
            showarrow=False,
            font=dict(size=11, color="#D32F2F"),
        )

        title = f"{state['situation']} | Ventilation {actions['ventilation']:.0f}% | Cooling {actions['cooling']:.0f}%"
        fig.add_annotation(x=6, y=7.45, text=title, showarrow=False, font=dict(size=15))

        fig.update_layout(
            height=430,
            xaxis=dict(visible=False, range=[-0.8, 13.4]),
            yaxis=dict(visible=False, range=[-0.6, 7.8]),
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )

        return fig


    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Room state visualization")
        fig_room = make_room_figure(state, actions, status_color)
        st.plotly_chart(fig_room, width="stretch", key="room_state_plot")


    with right:
        st.subheader("3. Digital twin control decision")

        def pct_bar(label, value, caption=""):
            st.write(f"**{label}: {value:.0f}%**")
            st.progress(int(clamp(value, 0, 100)))
            if caption:
                st.caption(caption)

        pct_bar("Ventilation", actions["ventilation"], "Removes CO₂ and stabilizes humidity.")
        pct_bar("Outside air intake", actions["outside_air_intake"])
        pct_bar("Air filtration", actions["filtration"], "Reduces PM10, VOC and formaldehyde.")
        pct_bar("Heating", actions["heating"])
        pct_bar("Cooling", actions["cooling"])
        pct_bar("Lighting", actions["lighting"])

        if actions["emergency_lights"] > 0:
            pct_bar("Emergency lights", actions["emergency_lights"])

        if actions["smoke_exhaust"] > 0:
            pct_bar("Smoke exhaust", actions["smoke_exhaust"])

        st.write(f"**Energy saving:** {'ON' if actions['energy_saving'] else 'OFF'}")
        st.write(f"**Alarm:** {actions['alarm_state']}")
        st.write(f"**Exits:** {actions['exits_state']}")

        with st.expander("Why did the system choose these actions?"):
            for note in actions["notes"]:
                st.write(f"- {note}")

        st.info(
            "After this step, these actions are applied to the room model. "
            "The changed room state becomes the next sensor input."
        )


    # ============================================================
    # EXPLANATION
    # ============================================================

    with st.expander("Why did the system act?"):
        for note in actions["notes"]:
            st.write(f"- {note}")


    # ============================================================
    # HISTORY UPDATE AND CHARTS
    # ============================================================

    history_row = {
        "step": step,
        "situation": state["situation"],
        "space_mode": state["space_mode"],
        "occupancy": state["occupancy"],
        "co2_ppm": state["co2_ppm"],
        "temperature_c": state["indoor_temperature_c"],
        "humidity_percent": state["relative_humidity_percent"],
        "tvoc_ug_m3": state["tvoc_ug_m3"],
        "battery_percent": state["battery_percent"],
        "comfort_score": state["comfort_score"],
        "ventilation": actions["ventilation"],
        "filtration": actions["filtration"],
        "heating": actions["heating"],
        "cooling": actions["cooling"],
        "lighting": actions["lighting"],
        "forecast_co2_without_action": forecast_no_action["co2_ppm"],
        "forecast_co2_after_action": forecast_after_action["co2_ppm"],
        "forecast_temp_without_action": forecast_no_action["indoor_temperature_c"],
        "forecast_temp_after_action": forecast_after_action["indoor_temperature_c"],
    }

    if not st.session_state.history or st.session_state.history[-1]["step"] != step:
        st.session_state.history.append(history_row)

    history = pd.DataFrame(st.session_state.history).tail(160)

    st.subheader("Closed-loop simulation history")

    tab_env, tab_actions = st.tabs(["Environment", "Actions"])

    with tab_env:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["step"], y=history["co2_ppm"], mode="lines", name="Actual CO₂"))
        fig.add_trace(go.Scatter(x=history["step"], y=history["forecast_co2_without_action"], mode="lines", name="Forecast without action", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=history["step"], y=history["forecast_co2_after_action"], mode="lines", name="Forecast after action", line=dict(dash="dot")))
        fig.update_layout(height=330, xaxis_title="Simulation step", yaxis_title="CO₂, ppm")
        st.plotly_chart(fig, width="stretch", key="co2_history_plot")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=history["step"], y=history["temperature_c"], mode="lines", name="Actual temperature"))
        fig2.add_trace(go.Scatter(x=history["step"], y=history["forecast_temp_without_action"], mode="lines", name="Forecast without action", line=dict(dash="dash")))
        fig2.add_trace(go.Scatter(x=history["step"], y=history["forecast_temp_after_action"], mode="lines", name="Forecast after action", line=dict(dash="dot")))
        fig2.update_layout(height=330, xaxis_title="Simulation step", yaxis_title="Temperature, °C")
        st.plotly_chart(fig2, width="stretch", key="temperature_history_plot")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=history["step"], y=history["comfort_score"], mode="lines", name="Comfort score"))
        fig3.add_trace(go.Scatter(x=history["step"], y=history["battery_percent"], mode="lines", name="Battery"))
        fig3.update_layout(height=330, xaxis_title="Simulation step", yaxis_title="Value")
        st.plotly_chart(fig3, width="stretch", key="comfort_battery_history_plot")

    with tab_actions:
        fig = go.Figure()
        for col, name in [
            ("ventilation", "Ventilation"),
            ("filtration", "Filtration"),
            ("heating", "Heating"),
            ("cooling", "Cooling"),
            ("lighting", "Lighting"),
        ]:
            fig.add_trace(go.Scatter(x=history["step"], y=history[col], mode="lines", name=name))

        fig.update_layout(height=380, xaxis_title="Simulation step", yaxis_title="Control level, %")
        st.plotly_chart(fig, width="stretch", key="actions_history_plot")




    if auto_run:
        st.session_state.state = next_state


render_simulation_frame()

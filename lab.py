
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
FIXED_UPDATE_INTERVAL = 0.3


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


def compute_ieq_metrics(state, thresholds):
    """
    IEQ comfort model inspired by the presentation:
    comfort is treated as the inverse of predicted dissatisfaction.

    The presentation frames the ideal indoor environment as comfortable,
    healthy and productive with the fewest dissatisfied occupants. It also
    separates thermal comfort, air quality and perceived indoor-environment
    quality. In this demo, we approximate that idea with:
    - thermal discomfort index, 0-3;
    - air-quality dissatisfaction, 0-100%;
    - humidity/crowding/energy penalties;
    - final comfort score = 100 - predicted dissatisfied percent.
    """

    temp = float(state["indoor_temperature_c"])
    humidity = float(state["relative_humidity_percent"])
    co2 = float(state["co2_ppm"])
    pm10 = float(state["pm10_ug_m3"])
    tvoc = float(state["tvoc_ug_m3"])
    hcho = float(state["formaldehyde_ug_m3"])
    battery = float(state["battery_percent"])
    occupancy_ratio = float(state["occupancy"]) / max(float(state["active_capacity"]), 1)

    # 1) Thermal discomfort index, 0-3.
    # Similar to the TPI / thermal discomfort curves in the presentation:
    # near preferred temperature discomfort is low; far from it dissatisfaction rises.
    preferred_temp = 22.5 if state.get("space_mode", "coworking") == "coworking" else 22.0
    temp_deviation = abs(temp - preferred_temp)

    if temp_deviation <= 0.5:
        thermal_discomfort = 0.0
    elif temp_deviation >= 4.0:
        thermal_discomfort = 3.0
    else:
        thermal_discomfort = ((temp_deviation - 0.5) / 3.5) * 3.0

    thermal_discomfort = clamp(thermal_discomfort, 0, 3)

    # Convert 0-3 thermal discomfort to approximate % dissatisfied.
    # Even near neutral, a small share of people can still be dissatisfied.
    thermal_dissatisfied = clamp(5 + (thermal_discomfort / 3) * 65, 0, 85)

    # 2) Air quality dissatisfaction.
    # The presentation shows dissatisfaction increasing with air pollution concentration.
    # CO2 effect:
    # < 1000 ppm: acceptable;
    # 1000-1400 ppm: stuffiness, sleepiness and concentration decline;
    # > 1400 ppm: strong productivity and attention risk.
    if co2 <= 1000:
        co2_dissatisfied = clamp((co2 - 700) / 300 * 20, 0, 20)
    elif co2 <= 1400:
        co2_dissatisfied = clamp(20 + (co2 - 1000) / 400 * 45, 20, 65)
    else:
        co2_dissatisfied = clamp(65 + (co2 - 1400) / 600 * 30, 65, 95)
    pm10_dissatisfied = clamp((pm10 - 10) / max(thresholds["pm10_warn"] * 2 - 10, 1) * 55, 0, 80)
    tvoc_dissatisfied = clamp((tvoc - 200) / max(thresholds["tvoc_crit"] - 200, 1) * 75, 0, 90)
    hcho_dissatisfied = clamp((hcho - 10) / max(thresholds["hcho_warn"] * 2 - 10, 1) * 50, 0, 75)

    air_quality_dissatisfied = max(
        co2_dissatisfied,
        0.6 * tvoc_dissatisfied + 0.25 * pm10_dissatisfied + 0.15 * hcho_dissatisfied,
    )

    # 3) Humidity dissatisfaction.
    if thresholds["humidity_min"] <= humidity <= thresholds["humidity_max"]:
        humidity_dissatisfied = 5
    elif humidity < thresholds["humidity_min"]:
        humidity_dissatisfied = clamp(5 + (thresholds["humidity_min"] - humidity) * 1.5, 5, 70)
    else:
        humidity_dissatisfied = clamp(5 + (humidity - thresholds["humidity_max"]) * 1.5, 5, 70)

    # 4) Crowding / shelter density dissatisfaction.
    crowding_dissatisfied = 0
    if occupancy_ratio > 0.70:
        crowding_dissatisfied = clamp((occupancy_ratio - 0.70) / 0.40 * 45, 0, 65)

    # 5) Battery penalty matters mostly in shelter mode because comfort is constrained by safety/autonomy.
    battery_dissatisfied = 0
    if battery < thresholds["battery_warn"]:
        battery_dissatisfied = clamp((thresholds["battery_warn"] - battery) * 1.2, 0, 60)

    # Weighting differs by mode.
    # Coworking: thermal comfort and productivity matter more.
    # Shelter: health/safety and air quality matter more.
    if state.get("space_mode", "coworking") == "shelter":
        dissatisfied_percent = (
            0.25 * thermal_dissatisfied
            + 0.45 * air_quality_dissatisfied
            + 0.10 * humidity_dissatisfied
            + 0.10 * crowding_dissatisfied
            + 0.10 * battery_dissatisfied
        )
    else:
        dissatisfied_percent = (
            0.45 * thermal_dissatisfied
            + 0.30 * air_quality_dissatisfied
            + 0.10 * humidity_dissatisfied
            + 0.10 * crowding_dissatisfied
            + 0.05 * battery_dissatisfied
        )

    dissatisfied_percent = clamp(dissatisfied_percent, 0, 100)
    comfort_score = clamp(100 - dissatisfied_percent, 0, 100)

    return {
        "comfort_score": comfort_score,
        "dissatisfied_percent": dissatisfied_percent,
        "thermal_discomfort": thermal_discomfort,
        "thermal_dissatisfied": thermal_dissatisfied,
        "air_quality_dissatisfied": air_quality_dissatisfied,
        "humidity_dissatisfied": humidity_dissatisfied,
        "crowding_dissatisfied": crowding_dissatisfied,
        "battery_dissatisfied": battery_dissatisfied,
    }


def compute_comfort_score(state, thresholds):
    return compute_ieq_metrics(state, thresholds)["comfort_score"]


def co2_effect_label(co2):
    co2 = float(co2)
    if co2 < 1000:
        return "acceptable"
    if co2 <= 1400:
        return "stuffiness / lower concentration"
    return "critical productivity risk"


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
            "co2_warn": 1000,
            "co2_crit": 1400,
            "temp_min": 17,
            "temp_max": 24,
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
        "co2_crit": 1400,
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
        "recovery_boost": False,
        "cycle_reset": False,
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
            base["outside_temperature_c"] = 29
            base["event_heat"] = 0.10
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
        # Full transition demo loop:
        # 1) green coworking;
        # 2) more people arrive and the room gets worse;
        # 3) the control system stabilizes it back to green;
        # 4) emergency mode starts and the same logic repeats in shelter mode.
        phase = step % 180

        if phase < 25:
            base["space_mode"] = "coworking"
            base["situation"] = "normal coworking"
            base["active_capacity"] = 60
            base["occupancy"] = int(14 + phase * 0.12)
            base["outside_humidity_percent"] = 45
            base["cycle_reset"] = phase < 3

        elif phase < 55:
            base["space_mode"] = "coworking"
            base["situation"] = "people arrive in coworking"
            base["active_capacity"] = 60
            base["occupancy"] = int(28 + (phase - 25) * 0.65)
            base["event_noise"] = 6
            base["event_tvoc"] = 8
            base["outside_humidity_percent"] = 48

        elif phase < 80:
            base["space_mode"] = "coworking"
            base["situation"] = "coworking stabilized after control"
            base["active_capacity"] = 60
            base["occupancy"] = 24
            base["outside_humidity_percent"] = 45
            base["recovery_boost"] = True

        elif phase < 95:
            base["space_mode"] = "shelter"
            base["situation"] = "emergency mode activated"
            base["active_capacity"] = 220
            base["occupancy"] = int(70 + (phase - 80) * 2.0)
            base["daylight_lux"] = min(base["daylight_lux"], 110)
            base["outside_humidity_percent"] = 50
            base["event_noise"] = 8

        elif phase < 115:
            base["space_mode"] = "shelter"
            base["situation"] = "stable shelter mode"
            base["active_capacity"] = 220
            base["occupancy"] = 105
            base["daylight_lux"] = 90
            base["outside_humidity_percent"] = 50
            base["recovery_boost"] = True

        elif phase < 145:
            base["space_mode"] = "shelter"
            base["situation"] = "shelter conditions worsen"
            base["active_capacity"] = 220
            base["occupancy"] = 165
            base["grid_available"] = False
            base["backup_power_only"] = True
            base["daylight_lux"] = 45
            base["outside_humidity_percent"] = 58
            base["event_noise"] = 12
            base["event_tvoc"] = 18

        else:
            base["space_mode"] = "shelter"
            base["situation"] = "shelter stabilized after control"
            base["active_capacity"] = 220
            base["occupancy"] = 95
            base["grid_available"] = True
            base["backup_power_only"] = False
            base["daylight_lux"] = 85
            base["outside_humidity_percent"] = 48
            base["recovery_boost"] = True

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

    ieq = compute_ieq_metrics(state, thresholds)
    state.update(ieq)
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
        notes.append("CO₂ is expected to enter the 1000–1400 ppm range → stuffiness, sleepiness and lower concentration risk. Ventilation increased proactively.")

    if co2_now > thresholds["co2_crit"] or co2_future > thresholds["co2_crit"]:
        actions["ventilation"] = 100
        notes.append("CO₂ is expected to exceed 1400 ppm → strong productivity and attention risk. Maximum ventilation selected.")

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
            "recovery_boost",
            "cycle_reset",
        ]:
            next_state[key] = value

    if inputs.get("cycle_reset", False):
        state = state.copy()
        state["co2_ppm"] = 620
        state["indoor_temperature_c"] = 22.2
        state["relative_humidity_percent"] = 45
        state["pm10_ug_m3"] = 14
        state["tvoc_ug_m3"] = 240
        state["formaldehyde_ug_m3"] = 14
        state["co_ppm"] = 0.4
        state["oxygen_percent"] = 20.9
        state["battery_percent"] = 100
        state["comfort_score"] = compute_comfort_score(state, get_thresholds(inputs["space_mode"]))

    people = inputs["occupancy"]
    ventilation = actions["ventilation"]
    filtration = actions["filtration"]
    cooling = actions["cooling"]
    heating = actions["heating"]

    # CO2 dynamics
    # The loop is tuned so that bad states are visible, but corrective ventilation
    # can bring the room back to an acceptable/green state within a few steps.
    co2_generation_factor = 0.60 if inputs["space_mode"] == "coworking" else 0.42
    removal_factor = 0.50 if not inputs.get("recovery_boost", False) else 0.72

    co2_generation = people * co2_generation_factor
    co2_removal = (ventilation / 100) * max(state["co2_ppm"] - 420, 0) * removal_factor

    if inputs.get("recovery_boost", False):
        co2_removal += max(state["co2_ppm"] - 650, 0) * 0.16

    next_state["co2_ppm"] = clamp(state["co2_ppm"] + co2_generation - co2_removal, 420, 2500)

    # Temperature dynamics
    # The room is treated as a basement / semi-basement space with high thermal inertia.
    # Therefore, even during heat waves, indoor temperature changes slowly and should not reach 35 °C.
    if inputs["space_mode"] == "shelter":
        thermal_inertia = 0.018
        people_heat_factor = 0.0012
        hvac_heat_effect = 0.35
        hvac_cool_effect = 0.45
        min_indoor_temp = 14.0
        max_indoor_temp = 26.0
    else:
        thermal_inertia = 0.025
        people_heat_factor = 0.0018
        hvac_heat_effect = 0.45
        hvac_cool_effect = 0.55
        min_indoor_temp = 16.0
        max_indoor_temp = 28.0

    people_heat = people * people_heat_factor
    outside_exchange = (ventilation / 100) * (inputs["outside_temperature_c"] - state["indoor_temperature_c"]) * thermal_inertia
    hvac_effect = (heating / 100) * hvac_heat_effect - (cooling / 100) * hvac_cool_effect
    next_state["indoor_temperature_c"] = clamp(
        state["indoor_temperature_c"] + people_heat + outside_exchange + hvac_effect + inputs["event_heat"],
        min_indoor_temp,
        max_indoor_temp,
    )

    # Humidity dynamics
    # Humidity can rise in a crowded shelter, but control + lower occupancy should bring it down again.
    humidity_generation = people * (0.0030 if inputs["space_mode"] == "coworking" else 0.0022)
    humidity_exchange = (ventilation / 100) * (inputs["outside_humidity_percent"] - state["relative_humidity_percent"]) * 0.060

    humidity_recovery = 0
    if inputs.get("recovery_boost", False):
        humidity_recovery = (state["relative_humidity_percent"] - 50) * 0.10

    max_humidity = 74 if inputs["space_mode"] == "shelter" else 68
    next_state["relative_humidity_percent"] = clamp(
        state["relative_humidity_percent"] + humidity_generation + humidity_exchange - humidity_recovery,
        30,
        max_humidity,
    )

    # Pollution dynamics
    recovery_multiplier = 1.6 if inputs.get("recovery_boost", False) else 1.0

    next_state["pm10_ug_m3"] = clamp(
        state["pm10_ug_m3"] + people * 0.012 + inputs["event_pm10"] - (filtration / 100) * state["pm10_ug_m3"] * 0.22 * recovery_multiplier,
        2,
        160,
    )

    next_state["tvoc_ug_m3"] = clamp(
        state["tvoc_ug_m3"] + people * 0.28 + inputs["event_tvoc"] - (filtration / 100) * state["tvoc_ug_m3"] * 0.16 * recovery_multiplier - (ventilation / 100) * state["tvoc_ug_m3"] * 0.06 * recovery_multiplier,
        50,
        1600,
    )

    next_state["formaldehyde_ug_m3"] = clamp(
        state["formaldehyde_ug_m3"] + people * 0.006 - (filtration / 100) * state["formaldehyde_ug_m3"] * 0.07 * recovery_multiplier,
        3,
        120,
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
    ieq = compute_ieq_metrics(next_state, thresholds)
    next_state.update(ieq)

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
    st.session_state.state_stack = []
    st.session_state.last_simulation_mode = simulation_mode
    st.session_state.is_running = True


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Simulation controls")

simulation_mode = st.sidebar.radio(
    "Simulation mode",
    ["Coworking only", "Shelter only", "Full transition"],
    index=2,
)

if "state" not in st.session_state or st.session_state.get("last_simulation_mode") != simulation_mode:
    reset_simulation(simulation_mode)

if "is_running" not in st.session_state:
    st.session_state.is_running = True

st.sidebar.caption(f"State changes automatically every {FIXED_UPDATE_INTERVAL:.1f} seconds.")

if st.session_state.is_running:
    if st.sidebar.button("Stop simulation", width="stretch"):
        st.session_state.is_running = False
        st.rerun()
else:
    if st.sidebar.button("Start simulation", width="stretch"):
        st.session_state.is_running = True
        st.rerun()

step_cols = st.sidebar.columns(2)

with step_cols[0]:
    if st.button("Step back", width="stretch"):
        st.session_state.manual_previous = True
        st.session_state.is_running = False

with step_cols[1]:
    if st.button("Next step", width="stretch"):
        st.session_state.manual_next = True
        st.session_state.is_running = False

if st.sidebar.button("Restart simulation", width="stretch"):
    reset_simulation(simulation_mode)
    st.rerun()

st.sidebar.caption("Step back returns to the previous simulated state. Next step advances the model once without autoplay.")



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


if st.session_state.pop("manual_previous", False):
    if st.session_state.get("state_stack"):
        st.session_state.state = st.session_state.state_stack.pop()
        if st.session_state.get("history"):
            st.session_state.history = st.session_state.history[:-1]
    st.rerun()

if st.session_state.pop("manual_next", False):
    manual_outputs = calculate_loop_outputs(st.session_state.state, simulation_mode)
    st.session_state.state_stack.append(st.session_state.state.copy())
    st.session_state.state = manual_outputs["next_state"]
    st.rerun()


# ============================================================
# FRAGMENTED UI
# ============================================================

run_every_value = f"{FIXED_UPDATE_INTERVAL}s" if st.session_state.is_running else None

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
                The demo loops: green coworking → worsening → control recovery → emergency shelter → worsening → control recovery.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # STEP 1 — CURRENT SENSOR INPUT
    # ============================================================

    st.subheader("1. Sensor input: current measured room state")

    sensor_cols = st.columns(8)

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

    with sensor_cols[6]:
        st.metric("Dissatisfied", f"{state.get('dissatisfied_percent', 100 - state['comfort_score']):.0f}%")

    with sensor_cols[7]:
        st.metric("CO₂ effect", co2_effect_label(state["co2_ppm"]))


    # ============================================================
    # STEPS 2–4 — CLOSED LOOP TABLE
    # ============================================================

    st.subheader("2–4. Forecast, decision and simulated result")

    closed_loop_table = pd.DataFrame(
        {
            "Indicator": ["CO₂", "CO₂ effect", "Temperature", "Humidity", "TVOC", "Battery", "Comfort", "Dissatisfied", "Thermal discomfort"],
            "1. Current sensor state": [
                f"{state['co2_ppm']:.0f} ppm",
                co2_effect_label(state["co2_ppm"]),
                f"{state['indoor_temperature_c']:.1f} °C",
                f"{state['relative_humidity_percent']:.0f}%",
                f"{state['tvoc_ug_m3']:.0f} μg/m³",
                f"{state['battery_percent']:.0f}%",
                f"{state['comfort_score']:.0f}/100",
                f"{state.get('dissatisfied_percent', 100 - state['comfort_score']):.0f}%",
                f"{state.get('thermal_discomfort', 0):.1f}/3",
            ],
            "2. Predicted in 10 min without action": [
                f"{forecast_no_action['co2_ppm']:.0f} ppm",
                co2_effect_label(forecast_no_action["co2_ppm"]),
                f"{forecast_no_action['indoor_temperature_c']:.1f} °C",
                f"{forecast_no_action['relative_humidity_percent']:.0f}%",
                f"{forecast_no_action['tvoc_ug_m3']:.0f} μg/m³",
                f"{forecast_no_action['battery_percent']:.0f}%",
                f"{forecast_no_action['comfort_score']:.0f}/100",
                f"{forecast_no_action.get('dissatisfied_percent', 100 - forecast_no_action['comfort_score']):.0f}%",
                f"{forecast_no_action.get('thermal_discomfort', 0):.1f}/3",
            ],
            "4. Predicted in 10 min after action": [
                f"{forecast_after_action['co2_ppm']:.0f} ppm",
                co2_effect_label(forecast_after_action["co2_ppm"]),
                f"{forecast_after_action['indoor_temperature_c']:.1f} °C",
                f"{forecast_after_action['relative_humidity_percent']:.0f}%",
                f"{forecast_after_action['tvoc_ug_m3']:.0f} μg/m³",
                f"{forecast_after_action['battery_percent']:.0f}%",
                f"{forecast_after_action['comfort_score']:.0f}/100",
                f"{forecast_after_action.get('dissatisfied_percent', 100 - forecast_after_action['comfort_score']):.0f}%",
                f"{forecast_after_action.get('thermal_discomfort', 0):.1f}/3",
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
        "dissatisfied_percent": state.get("dissatisfied_percent", 100 - state["comfort_score"]),
        "thermal_discomfort": state.get("thermal_discomfort", 0),
        "air_quality_dissatisfied": state.get("air_quality_dissatisfied", 0),
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




    if st.session_state.is_running:
        st.session_state.state_stack.append(state.copy())
        if len(st.session_state.state_stack) > 500:
            st.session_state.state_stack = st.session_state.state_stack[-500:]
        st.session_state.state = next_state


render_simulation_frame()

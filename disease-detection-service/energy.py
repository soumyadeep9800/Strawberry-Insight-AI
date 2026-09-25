REFERENCE_POWER_WATTS = 15.0
#So if the CPU were consuming 15 W continuously
# Carbon intensity in grams CO2e per kWh.
# Set this according to the electricity/grid factor you choose
# for your project methodology.
#15 W × actual inference time = estimated energy for that particular prediction.
CARBON_INTENSITY_G_PER_KWH = 700.0


def calculate_energy(inference_time_ms: float) -> float:
    """
    Estimate energy consumption for one inference.

    Energy (J) = Power (W) × Time (seconds)
    """
    inference_time_seconds = inference_time_ms / 1000

    return REFERENCE_POWER_WATTS * inference_time_seconds


def calculate_carbon_footprint(energy_joules: float) -> float:
    """
    Estimate carbon footprint for one inference.

    CO2e (g) = Energy (kWh) × Carbon intensity (gCO2e/kWh)
    """
    energy_kwh = energy_joules / 3_600_000

    return energy_kwh * CARBON_INTENSITY_G_PER_KWH
import onnx
import numpy as np
import time

init_temperature = 10
final_temperature_step = 50
heightened_final_step = 90
heightened_temp_min = .1

for step in range(100):
    temp = max(1, 1 + init_temperature * (final_temperature_step - step) / final_temperature_step)
    if temp == 1 and step > final_temperature_step and heightened_final_step and heightened_final_step != 1:
        # Once the temperature passes (1) it enters an inverted curve to match the linear curve from above.
        # without this, the attention specificity "spikes" incredibly fast in the last few iterations.
        h_steps_total = heightened_final_step - final_temperature_step
        h_steps_current = min(step - final_temperature_step, h_steps_total)
        # The "gap" will represent the steps that need to be traveled as a linear function.
        h_gap = 1 / heightened_temp_min
        temp = h_gap * h_steps_current / h_steps_total
        # Invert temperature to represent reality on this side of the curve
        temp = 1 / temp
    print("%i: %f" % (step, temp))
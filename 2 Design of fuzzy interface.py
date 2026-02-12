import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import networkx as nx

# [cite_start]1. Variables [cite: 9, 11, 13]
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
food = ctrl.Antecedent(np.arange(0, 11, 1), 'food')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# [cite_start]2. Membership Functions [cite: 17, 21, 25]
service.automf(3, names=['poor', 'acceptable', 'excellent'])
food['bad'] = fuzz.trapmf(food.universe, [0, 0, 1, 3])
food['decent'] = fuzz.trimf(food.universe, [1, 5, 9])
food['great'] = fuzz.trapmf(food.universe, [7, 9, 10, 10])
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# [cite_start]3. Rules [cite: 34-37]
rule1 = ctrl.Rule(service['poor'] | food['bad'], tip['low'])
rule2 = ctrl.Rule(service['acceptable'], tip['medium'])
rule3 = ctrl.Rule(service['excellent'] & food['great'], tip['high'])

# [cite_start]4. System [cite: 40-42]
tip_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tip_ctrl)

# [cite_start]5. Inputs [cite: 46-47]
tipping.input['service'] = 6.5
tipping.input['food'] = 9.8
tipping.compute()

print(f"--- Lab 2 Result ---")
print(f"Recommended Tip: {tipping.output['tip']:.2f}%")
tip.view(sim=tipping)
plt.show()
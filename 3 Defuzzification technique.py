import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def demonstrate_defuzzification(universe, aggregated_mf):
    # 1. Centroid (CoG/CoM) - Center of gravity [cite: 75, 76]
    cog = fuzz.defuzz(universe, aggregated_mf, 'centroid')

    # 2. Bisector (BoA) - Divides area into two equal halves [cite: 78, 79]
    boa = fuzz.defuzz(universe, aggregated_mf, 'bisector')

    # 3. Mean of Maximum (MoM) - Average of the highest points [cite: 81, 82]
    mom = fuzz.defuzz(universe, aggregated_mf, 'mom')

    # 4. Smallest of Maximum (SoM) - Smallest value with max height [cite: 84, 85]
    som = fuzz.defuzz(universe, aggregated_mf, 'som')

    # 5. Largest of Maximum (LoM) - Largest value with max height [cite: 87, 88]
    lom = fuzz.defuzz(universe, aggregated_mf, 'lom')

    # --- Print Results --- 
    print("--- Defuzzification Results ---")
    print(f"Centroid (CoG):        {cog:.4f}")
    print(f"Bisector (BoA):        {boa:.4f}")
    print(f"Mean of Maximum (MoM): {mom:.4f}")
    print(f"Smallest of Max (SoM): {som:.4f}")
    print(f"Largest of Max (LoM):  {lom:.4f}")

    # --- Plot Visualization --- [cite: 98-112]
    plt.figure(figsize=(10, 6))
    plt.plot(universe, aggregated_mf, 'b', linewidth=2.5, label='Aggregated Fuzzy Set')

    plt.axvline(cog, color='r', linestyle='--', label=f'Centroid ({cog:.2f})')
    plt.axvline(boa, color='g', linestyle='-.', label=f'Bisector ({boa:.2f})')
    plt.plot([mom, mom], [0, 1.0], 'k:', label=f'MoM ({mom:.2f})') 
    plt.plot([som, som], [0, 1.0], 'c:', label=f'SoM ({som:.2f})')
    plt.plot([lom, lom], [0, 1.0], 'm:', label=f'LoM ({lom:.2f})')

    plt.title('Comparison of Defuzzification Techniques')
    plt.ylabel('Membership Degree')
    plt.xlabel('Output Universe')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 1. Define the Universe and Fuzzy Set --- [cite: 113-115]
X = np.arange(0, 26, 0.1)

# Create two shapes (trapezoid and triangle) to represent output [cite: 120-121]
mf_1 = fuzz.trapmf(X, [0, 5, 8, 11])
mf_2 = fuzz.trimf(X, [9, 15, 25])

# Combine them into one "Aggregated" set [cite: 124]
aggregated_mf = np.fmax(mf_1 * 0.7, mf_2 * 1.0) 

# Run the demonstration [cite: 126]
demonstrate_defuzzification(X, aggregated_mf)
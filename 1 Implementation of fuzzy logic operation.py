import numpy as np

# Data
A = np.array([1.0, 0.8, 0.4, 0.1, 0.0])
B = np.array([0.0, 0.1, 0.3, 0.7, 1.0])

print("--- Lab 1 Results ---")
print(f"UNION (OR): {np.maximum(A, B)}")
print(f"INTERSECTION (AND): {np.minimum(A, B)}")
print(f"COMPLEMENT (NOT A): {1 - A}")
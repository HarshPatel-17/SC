import numpy as np

def mcp_activation(net_input, threshold):
    return 1 if net_input >= threshold else 0

def mcp_neuron(inputs, weights, threshold):
    inputs = np.array(inputs)
    weights = np.array(weights)
    net_input = np.dot(inputs, weights) # Calculate weighted sum [cite: 147]
    output = mcp_activation(net_input, threshold)
    return net_input, output

def implement_or_gate():
    print("\n--- Implementing Logical OR Gate ---")
    weights = [1, 1] # Equal weights [cite: 158]
    threshold = 1    # Threshold for OR [cite: 159]
    test_cases = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
    for inputs, expected in test_cases:
        _, output = mcp_neuron(inputs, weights, threshold)
        print(f"Input: {inputs}, Output: {output}, Expected: {expected}")

def implement_not_gate():
    print("\n--- Implementing Logical NOT Gate ---")
    weights = [-1]   # Negative weight to inhibit [cite: 177]
    threshold = 0 
    test_cases = [([0], 1), ([1], 0)]
    for inputs, expected in test_cases:
        _, output = mcp_neuron(inputs, weights, threshold)
        print(f"Input: {inputs}, Output: {output}, Expected: {expected}")

implement_or_gate()
implement_not_gate()
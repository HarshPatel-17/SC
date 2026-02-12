import numpy as np
import matplotlib.pyplot as plt

# --- 1. Activation and Derivative Functions (Sigmoid) ---
def sigmoid(x):
    """Sigmoid activation function: 1 / (1 + e^(-x))"""
    x = np.clip(x, -500, 500) # Prevents math errors for large numbers [cite: 293]
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    """Derivative used for updating weights during backpropagation [cite: 298]"""
    return output * (1 - output)

# --- 2. Define the Multilayer Perceptron Class ---
class MLP_Backpropagation:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.2, max_epochs=10000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # Initialize Weights and Biases randomly [cite: 310-315]
        self.W_ih = np.random.uniform(low=-0.5, high=0.5, size=(input_size, hidden_size))
        self.b_h = np.zeros((1, hidden_size))
        self.W_ho = np.random.uniform(low=-0.5, high=0.5, size=(hidden_size, output_size))
        self.b_o = np.zeros((1, output_size))
        self.errors = []

    def forward_pass(self, X):
        """Moves data forward through the layers [cite: 317-328]"""
        self.net_h = np.dot(X, self.W_ih) + self.b_h
        self.out_h = sigmoid(self.net_h)
        self.net_o = np.dot(self.out_h, self.W_ho) + self.b_o
        self.out_o = sigmoid(self.net_o)
        return self.out_o

    def backward_pass(self, X, y, out_o, out_h):
        """Core Backpropagation: updates weights based on error [cite: 329-354]"""
        error_o = y - out_o
        d_o = error_o * sigmoid_derivative(out_o) # Output layer delta

        error_h = d_o.dot(self.W_ho.T)
        d_h = error_h * sigmoid_derivative(out_h) # Hidden layer delta

        # Update weights and biases using Gradient Descent [cite: 345-353]
        self.W_ho += out_h.T.dot(d_o) * self.learning_rate
        self.b_o += np.sum(d_o, axis=0, keepdims=True) * self.learning_rate
        self.W_ih += X.T.dot(d_h) * self.learning_rate
        self.b_h += np.sum(d_h, axis=0, keepdims=True) * self.learning_rate
        return np.mean(error_o**2)

    def train(self, X_train, y_train):
        print(f"--- Training MLP (Hidden Neurons: 4, Epochs: {self.max_epochs}) ---")
        for epoch in range(self.max_epochs):
            out_o = self.forward_pass(X_train)
            mse = self.backward_pass(X_train, y_train, out_o, self.out_h)
            self.errors.append(mse)
            if (epoch + 1) % 2000 == 0:
                print(f"Epoch {epoch + 1}, Mean Squared Error: {mse:.6f}")

# --- 3. Data for XOR Gate ---
# XOR: Output is 1 only if the inputs are different [cite: 369-383]
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# --- 4. Run and Test Model ---
mlp = MLP_Backpropagation(input_size=2, hidden_size=4, output_size=1)
mlp.train(X_train, y_train)

print("\n--- Testing XOR Predictions ---")
predictions = mlp.forward_pass(X_train)
for inputs, prediction, expected in zip(X_train, predictions, y_train):
    # Use 0.5 as the threshold to decide 0 or 1 [cite: 401]
    predicted_class = 1 if prediction[0] >= 0.5 else 0
    print(f"Input: {inputs}, Output: {prediction[0]:.4f}, Predicted: {predicted_class}, Expected: {expected[0]}")

# Visualize Error History [cite: 407-413]
plt.plot(mlp.errors)
plt.title('MLP Training Error (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
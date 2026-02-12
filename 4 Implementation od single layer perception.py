import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Activation Function (Step Function) ---
def step_function(weighted_sum):
    return 1 if weighted_sum >= 0 else 0


# --- 2. Define the Perceptron Class ---
class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=50):
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=num_inputs)
        self.bias = np.random.uniform(low=-0.5, high=0.5, size=1)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.errors = []

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return step_function(weighted_sum)

    def train(self, training_inputs, labels):
        print(f"--- Training Perceptron (Rate: {self.learning_rate}) ---")
        for epoch in range(self.max_epochs):
            total_error = 0

            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += abs(error)

                if error != 0:
                    self.weights += self.learning_rate * error * inputs
                    self.bias += self.learning_rate * error

            self.errors.append(total_error)

            if total_error == 0:
                print(f"Converged at Epoch {epoch + 1}")
                break


# --- 3. Data for AND Gate ---
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_train = np.array([0, 0, 0, 1])


# --- 4. Run Model ---
perceptron = Perceptron(num_inputs=2, learning_rate=0.1)
perceptron.train(X_train, y_train)


# --- 5. Test Model ---
print("\n--- Testing Model ---")
for inputs, expected in zip(X_train, y_train):
    prediction = perceptron.predict(inputs)
    print(f"Input: {inputs} -> Predicted: {prediction}, Expected: {expected}")


# --- 6. Plot Training Errors ---
plt.plot(perceptron.errors, marker='o')
plt.title("Perceptron Training Error")
plt.xlabel("Epoch")
plt.ylabel("Number of Errors")
plt.grid(True)
plt.show()

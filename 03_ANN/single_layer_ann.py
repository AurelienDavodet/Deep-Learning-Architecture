import numpy as np


class SingleLayerANN:
    def __init__(self, num_inputs: int, num_outputs: int, learning_rate: float = 0.1):
        """
        Initializes a single-layer ANN.

        Args:
            num_inputs (int): Number of input features.
            num_outputs (int): Number of output neurons (number of classes).
            learning_rate (float): Learning rate for weight updates.
        """
        self.weights = np.random.rand(num_inputs, num_outputs)  # Weight matrix
        self.biases = np.random.rand(num_outputs)  # Bias vector
        self.learning_rate = learning_rate

    def activation_function(self, x: float) -> int:
        """
        Step activation function for binary output.

        Args:
            x (float): Input value.

        Returns:
            int: 1 if x >= 0, else 0.
        """
        return 1 if x >= 0 else 0

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            inputs (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted binary outputs for each output neuron.
        """
        weighted_sum = np.dot(inputs, self.weights) + self.biases
        return np.vectorize(self.activation_function)(weighted_sum)

    def train(
        self, training_data: np.ndarray, labels: np.ndarray, epochs: int = 1000
    ) -> None:
        """
        Trains the single-layer ANN using the provided data.

        Args:
            training_data (np.ndarray): Input data.
            labels (np.ndarray): Target labels.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            for inputs, expected_output in zip(training_data, labels):
                predicted_output = self.predict(inputs)
                error = expected_output - predicted_output
                # Update weights and biases
                self.weights += self.learning_rate * np.outer(inputs, error)
                self.biases += self.learning_rate * error

            # Print progress
            if epoch % 100 == 0:
                loss = np.mean((labels - self.predict(training_data)) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def evaluate(self, test_data: np.ndarray) -> np.ndarray:
        """
        Evaluates the model on test data.

        Args:
            test_data (np.ndarray): Input features.

        Returns:
            np.ndarray: Predictions for the test data.
        """
        return self.predict(test_data)

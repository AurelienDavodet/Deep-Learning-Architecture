import numpy as np


class Perceptron:
    def __init__(self, num_inputs: int, learning_rate: float = 0.1):
        """
        Initializes the Perceptron with a specified number of inputs and learning rate.

        Args:
            num_inputs (int): The number of inputs for the perceptron.
            learning_rate (float): The learning rate for weight updates. Default is 0.1.
        """
        self.weights: np.ndarray = np.random.rand(num_inputs + 1)  # +1 for the bias
        self.learning_rate: float = learning_rate

    def activation_function(self, x: float) -> int:
        """
        The step activation function for binary classification.

        Args:
            x (float): The input value (weighted sum).

        Returns:
            int: Returns 1 if x >= 0, otherwise 0.
        """
        return 1 if x >= 0 else 0

    def predict(self, inputs: np.ndarray) -> int:
        """
        Predicts the output for given inputs using the perceptron model.

        Args:
            inputs (np.ndarray): Input features as a 1D numpy array.

        Returns:
            int: Predicted output, either 0 or 1.
        """
        # Calculate the weighted sum of inputs plus bias
        total_activation: float = np.dot(inputs, self.weights[:-1]) + self.weights[-1]
        return self.activation_function(total_activation)

    def train(
        self, training_data: np.ndarray, labels: np.ndarray, epochs: int = 10
    ) -> None:
        """
        Trains the perceptron on the provided training data and labels.

        Args:
            training_data (np.ndarray): Training examples as a 2D numpy array where each row is an example.
            labels (np.ndarray): Labels for each training example as a 1D numpy array.
            epochs (int): The number of training epochs. Default is 10.
        """
        for epoch in range(epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction

                # Update weights and bias based on error
                self.weights[:-1] += (
                    self.learning_rate * error * inputs
                )  # Update weights
                self.weights[-1] += self.learning_rate * error  # Update bias weight

            # Optionally print the training progress
            print(f"Epoch {epoch + 1}/{epochs} completed.")

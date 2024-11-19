import numpy as np


class MultiLayerANN:
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        learning_rate: float = 0.1,
    ):
        """
        Initializes a multi-layer ANN with one hidden layer.

        Args:
            num_inputs (int): Number of input features.
            num_hidden (int): Number of neurons in the hidden layer.
            num_outputs (int): Number of neurons in the output layer.
            learning_rate (float): Learning rate for weight updates.
        """
        self.weights_input_hidden = np.random.rand(num_inputs, num_hidden)
        self.bias_hidden = np.random.rand(num_hidden)
        self.weights_hidden_output = np.random.rand(num_hidden, num_outputs)
        self.bias_output = np.random.rand(num_outputs)
        self.learning_rate = learning_rate

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid applied element-wise.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid function.

        Args:
            x (np.ndarray): Sigmoid output.

        Returns:
            np.ndarray: Derivative of sigmoid.
        """
        return x * (1 - x)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            inputs (np.ndarray): Input features.

        Returns:
            np.ndarray: Output of the network.
        """
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = (
            np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        )
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(
        self, inputs: np.ndarray, expected_output: np.ndarray, output: np.ndarray
    ) -> None:
        """
        Backpropagation for weight updates.

        Args:
            inputs (np.ndarray): Input features.
            expected_output (np.ndarray): Target labels.
            output (np.ndarray): Predicted output.
        """
        error_output = expected_output - output
        delta_output = error_output * self.sigmoid_derivative(output)

        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += (
            self.hidden_output.T.dot(delta_output) * self.learning_rate
        )
        self.bias_output += np.sum(delta_output, axis=0) * self.learning_rate
        self.weights_input_hidden += inputs.T.dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0) * self.learning_rate

    def train(
        self, training_data: np.ndarray, labels: np.ndarray, epochs: int = 1000
    ) -> None:
        """
        Trains the multi-layer ANN using backpropagation.

        Args:
            training_data (np.ndarray): Input data.
            labels (np.ndarray): Target labels.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            output = self.forward(training_data)
            self.backward(training_data, labels, output)
            if epoch % 100 == 0:
                loss = np.mean(np.square(labels - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.

        Args:
            inputs (np.ndarray): Input features.

        Returns:
            np.ndarray: Network predictions.
        """
        return self.forward(inputs)

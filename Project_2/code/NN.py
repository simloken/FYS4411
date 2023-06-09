import tensorflow as tf
import numpy as np

class QuantumEnergyModel(tf.keras.Model):
    """
    A neural network model for estimating the energy of a quantum mechanical system.

    This model is designed to estimate the energy of a quantum mechanical system based on the interact value.

    Attributes:
        dense1: A dense layer with 32 units and ReLU activation function.
        dense2: A dense layer with 16 units and ReLU activation function.
        dense3: A dense layer with 8 units and ReLU activation function.
        dense4: A dense layer with 1 unit and linear activation function.

    Methods:
        call(inputs): Executes the forward pass of the model.

    """
    def __init__(self):
        super(QuantumEnergyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(8, activation='relu')
        self.dense4 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        """
        Inherent component of Tensorflow that handles the forward pass
    
        Args:
            inputs: Input data to the model.

        Returns:
            The output of the model after the forward pass.
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return output



def NN(interact):
    """
   Trains a QuantumEnergyModel neural network and estimates the energy of a quantum mechanical system.

   Args:
       interact: A boolean value indicating the interaction status of the system.
   """
    
    num_samples = 1000
    inputs = np.random.choice([True, False], size=num_samples)
    true_energy = np.where(inputs, 3.0, 2.0)
    inputs = np.reshape(inputs, (-1, 1))
    
    model = QuantumEnergyModel()
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(inputs, true_energy, epochs=20, batch_size=32, verbose=0)
    
    predicted_energies = []
    for i in range(100):
        new_input = np.array([[interact]])
        predicted_energies.append(model.predict(new_input, verbose=0))
    blocking_results = blocking_analysis(predicted_energies)
    analytical_result = analytical_energy(interact, 2, 2)
    
    print('Blocking analysis: %.2f +- %g' % (blocking_results[0], blocking_results[1]))
    
    print('Analytical result: %.2f' % analytical_result) 
    
def analytical_energy(interact, particles, dimensions):
    """
    Calculates the analytical energy of a system.

    Args:
        interact (bool): A flag indicating whether to include interaction energy or not.
        particles (int): The number of particles in the system.
        dimensions (int): The number of dimensions in the system.

    Returns:
        float: The analytical energy of the system.

    """
    if interact:
        if particles == 2 and dimensions == 2:
            return 3
        else:
            return 'Unknown'
        
    return 0.5 * particles * dimensions
    
def blocking_analysis(samples):
    """
    Performs blocking analysis on a set of samples to estimate the statistical error.

    Args:
        samples (list): A list of samples generated from a Monte Carlo simulation.

    Returns:
        float: The average mean value of the samples.
        float: The average statistical error of the samples.

    """
    n = len(samples)
    m = int(np.log2(n))

    means = [np.mean(samples)]
    errors = [np.std(samples) / np.sqrt(n)]

    for i in range(m):
        new_samples = []
        for j in range(0, n - 1, 2):
            new_samples.append((samples[j] + samples[j+1]) / 2)
        if n % 2 != 0:
            new_samples.append(samples[-1])
        samples = np.array(new_samples)
        n = len(samples)
        means.append(np.mean(samples))
        errors.append(np.std(samples) / np.sqrt(n))
        

    return np.mean(means), np.mean(errors)


NN(True)
NN(False)
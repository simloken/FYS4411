import numpy as np

class RBM:
    """
    Restricted Boltzmann Machine (RBM) implementation.

    Args:
        n_visible (int): Number of visible units.
        n_hidden (int): Number of hidden units.

    Attributes:
        n_visible (int): Number of visible units.
        n_hidden (int): Number of hidden units.
        weights (numpy.ndarray): Weight matrix with shape (n_visible, n_hidden).
        visible_bias (numpy.ndarray): Visible bias vector with shape (n_visible,).
        hidden_bias (numpy.ndarray): Hidden bias vector with shape (n_hidden,).
        energyList (None or list): List to store energy values during training.
        state (None or numpy.ndarray): Current state of the RBM.

    Methods:
        persistent_contrastive_divergence(N_CHAINS, LEARNING_RATE, visible, persistent_chain, PERSISTENT_CHAIN_RATE):
            Performs persistent contrastive divergence training on the RBM.
        
        gradient_descent(LEARNING_RATE, visible):
            Performs gradient descent training on the RBM.
        
        sample_hidden(visible):
            Samples the hidden units given the visible units.
        
        sample_visible(hidden):
            Samples the visible units given the hidden units.
        
        hidden_probabilities(visible):
            Calculates the probabilities of the hidden units being activated given the visible units.
        
        visible_probabilities(hidden):
            Calculates the probabilities of the visible units being activated given the hidden units.
        
        gibbs_sampling(visible):
            Performs Gibbs sampling by sampling the hidden units and then the visible units given the hidden units.
        
        importance_sampling(visible):
            Performs importance sampling by sampling the hidden units and calculating the probabilities of the visible units.
        
        contrastive_divergence(N_CHAINS, LEARNING_RATE, visible):
            Performs Contrastive Divergence (CD) training on the RBM.
    """
    def __init__(self, n_visible, n_hidden, sample):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weights = np.random.randn(n_visible, n_hidden) * 0.1
        self.visible_bias = np.random.randn(n_visible) * 0.1
        self.hidden_bias = np.random.randn(n_hidden) * 0.1
        
        if sample=='metropolis':
            self.sampler = self.gibbs_sampling
        else:
            self.sampler = self.importance_sampling
        
        self.energyList = None
        self.state = None
        
    def sample_hidden(self, visible):
        """
        Samples the hidden units given the visible units.
    
        Args:
            visible (numpy.ndarray): Visible units with shape (batch_size, visible_units).
    
        Returns:
            numpy.ndarray: Sampled hidden units with shape (batch_size, hidden_units).
    
        """
        hidden_prob = self.hidden_probabilities(visible)
        hidden_state = np.random.binomial(1, hidden_prob)
        return hidden_state

    def sample_visible(self, hidden):
        """
        Samples the visible units given the hidden units.
    
        Args:
            hidden (numpy.ndarray): Hidden units with shape (batch_size, hidden_units).
    
        Returns:
            numpy.ndarray: Sampled visible units with shape (batch_size, visible_units).
    
        """
        visible_prob = self.visible_probabilities(hidden)
        visible_state = np.random.binomial(1, visible_prob)
        return visible_state

    def hidden_probabilities(self, visible):
        """
        Calculates the probabilities of the hidden units being activated given the visible units.
    
        Args:
            visible (numpy.ndarray): Visible units with shape (batch_size, visible_units).
    
        Returns:
            numpy.ndarray: Probabilities of the hidden units being activated with shape (batch_size, hidden_units).
    
        """
        hidden_activations = np.dot(visible, self.weights) + self.hidden_bias
        hidden_prob = sigmoid(hidden_activations)
        return hidden_prob

    def visible_probabilities(self, hidden):
        """
        Calculates the probabilities of the visible units being activated given the hidden units.
    
        Args:
            hidden (numpy.ndarray): Hidden units with shape (batch_size, hidden_units).
    
        Returns:
            numpy.ndarray: Probabilities of the visible units being activated with shape (batch_size, visible_units).
            
        """
        visible_activations = np.dot(hidden, self.weights.T) + self.visible_bias
        visible_prob = sigmoid(visible_activations)
        return visible_prob

    def gibbs_sampling(self, visible):
        """
        Performs Gibbs sampling by sampling the hidden units and then sampling the visible units given the hidden units.
    
        Args:
            visible (numpy.ndarray): Visible units with shape (batch_size, visible_units).
    
        Returns:
            numpy.ndarray: Sampled visible units with shape (batch_size, visible_units).
    
        """
        hidden_state = self.sample_hidden(visible)
        visible_state = self.sample_visible(hidden_state)
        return visible_state
    
    def importance_sampling(self, visible):
        """
        Performs importance sampling by sampling the hidden units and calculating the probabilities of the visible units.
    
        Args:
            visible (numpy.ndarray): Visible units with shape (batch_size, visible_units).
    
        Returns:
            numpy.ndarray: Probabilities of the visible units being activated with shape (batch_size, visible_units).

        """
        hidden_prob = self.hidden_probabilities(visible)
        hidden_state = np.random.binomial(1, hidden_prob)
        visible_prob = self.visible_probabilities(hidden_state)
        return visible_prob

    def contrastive_divergence(self, N_CHAINS, LEARNING_RATE, visible):
        """
       Performs Contrastive Divergence (CD) training on the Restricted Boltzmann Machine.
    
       Args:
           N_CHAINS (int): Number of chains to sample.
           LEARNING_RATE (float): Learning rate for updating the weights and biases.
           visible (numpy.ndarray): Input data with shape (batch_size, visible_units). 
       """
        positive_hidden = self.sample_hidden(visible)
        positive_gradient = np.dot(visible.T, positive_hidden)
    
        chain_end = visible
        for _ in range(N_CHAINS):
            chain_end = self.sampler(chain_end)
    
        negative_hidden = self.sample_hidden(chain_end)
        negative_gradient = np.dot(chain_end.T, negative_hidden)
    
        self.weights += LEARNING_RATE * (positive_gradient - negative_gradient)
        self.visible_bias += LEARNING_RATE * np.mean(visible - chain_end, axis=0)
        self.hidden_bias += LEARNING_RATE * np.mean(positive_hidden - negative_hidden, axis=0)
        
    def persistent_contrastive_divergence(self, N_CHAINS, LEARNING_RATE, visible, persistent_chain, PERSISTENT_CHAIN_RATE):
        """
        Performs Persistent Contrastive Divergence (PCD) training on the Restricted Boltzmann Machine.
    
        Args:
            N_CHAINS (int): Number of chains to sample.
            LEARNING_RATE (float): Learning rate for updating the weights.
            visible (numpy.ndarray): Input data with shape (batch_size, visible_units).
            persistent_chain (numpy.ndarray): Persistent chain with shape (batch_size, visible_units).
            PERSISTENT_CHAIN_RATE (float): Weight rate for updating the persistent chain.
    
        """
        positive_hidden = self.sample_hidden(visible)
        positive_gradient = np.dot(visible.T, positive_hidden)
        
        chain_end = visible
        for _ in range(N_CHAINS):
            chain_end = self.sampler(chain_end)
        
        negative_hidden = self.sample_hidden(chain_end)
        negative_gradient = np.dot(chain_end.T, negative_hidden)
        
        self.weights += LEARNING_RATE * (positive_gradient - negative_gradient)
        self.visible_bias += LEARNING_RATE * np.mean(visible - chain_end, axis=0)
        self.hidden_bias += LEARNING_RATE * np.mean(positive_hidden - negative_hidden, axis=0)
        
        persistent_hidden = self.sample_hidden(persistent_chain)
        persistent_visible = self.sample_visible(persistent_hidden)
        self.weights += LEARNING_RATE * PERSISTENT_CHAIN_RATE * np.dot(persistent_chain.T, persistent_hidden)
        self.visible_bias += LEARNING_RATE * PERSISTENT_CHAIN_RATE * np.mean(persistent_chain - persistent_visible, axis=0)
        self.hidden_bias += LEARNING_RATE * PERSISTENT_CHAIN_RATE * np.mean(persistent_hidden - self.sample_hidden(persistent_visible), axis=0)
        
    def gradient_descent(self, LEARNING_RATE, visible):
        """
        Performs gradient descent (GD) training on the Restricted Boltzmann Machine.
    
        Args:
            LEARNING_RATE (float): Learning rate for updating the weights and biases.
            visible (numpy.ndarray): Input data with shape (batch_size, visible_units).
        """
        hidden_state = self.sample_hidden(visible)
        positive_gradient = np.dot(visible.T, hidden_state)
        
        visible_recon = self.sample_visible(hidden_state)
        hidden_recon = self.sample_hidden(visible_recon)
        negative_gradient = np.dot(visible_recon.T, hidden_recon)
        
        self.weights += LEARNING_RATE * (positive_gradient - negative_gradient)
        self.visible_bias += LEARNING_RATE * np.mean(visible - visible_recon, axis=0)
        self.hidden_bias += LEARNING_RATE * np.mean(hidden_state - hidden_recon, axis=0)
        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

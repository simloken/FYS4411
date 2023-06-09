from RBM import RBM
import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import pickle


# Constants
N_SAMPLES = 5000 # Number of Monte Carlo samples
N_CONVERGE = 500000 # Number of cycles to converge with CD
N_HIDDEN = 16  # Number of hidden units in RBM
N_VISIBLE = 4  # Number of visible units in RBM (two particles with two dimensions)
LEARNING_RATE = 0.005 # Learning rate for RBM
N_CHAINS = 100  # Number of Markov chains for Metropolis sampling
N_SWEEPS = 100  # Number of sweeps for each chain
N_SCALE = 10 # Some weird value that somehow makes the code work perfectly. Leave this at a multiple of 10 or CD and GD may break and not converge as expected. PCD always works, even with N_SCALE = 1.
omega = 1 # 1 to get a.u.

def local_energy(positions, interact):
    """
    Calculates the local energy of a system given the positions of particles.

    Args:
        positions (ndarray): An array of shape (num_particles, num_dimensions) containing the positions of particles.
        interact (bool): A flag indicating whether to include interaction energy or not.

    Returns:
        float: The local energy of the system.

    """
    kinetic_energy = np.sum(-0.5 * np.sum(np.gradient(positions, axis=1)**2, axis=1))
    potential_energy = 0.5 * omega**2 * np.sum(np.sum(positions**2, axis=1))
    if interact:
        interact_energy = interaction_energy(positions)
    else:
        interact_energy = 0
    return kinetic_energy + potential_energy + interact_energy

def interaction_energy(positions):
    """
    Calculates the interaction energy between particles in a system.

    Args:
        positions (ndarray): An array of shape (num_particles, num_dimensions) containing the positions of particles.

    Returns:
        float: The interaction energy between particles.

    """
    num_particles, num_dimensions = positions.shape
    energy = 0.0
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            distance = np.sqrt(np.sum((positions[i] - positions[j])**2))
            energy += 1.0 / max(distance, 1e-8)
            
    return energy

def metropolis_sampling(rbm, initial_state, interact, n_samples, n_sweeps, burn_in):
    """
    Performs Metropolis sampling using a Restricted Boltzmann Machine (RBM).

    Args:
        rbm (RBM): An instance of the Restricted Boltzmann Machine class.
        initial_state (ndarray): An array representing the initial state of the system.
        interact (bool): A flag indicating whether to include interaction energy or not.
        n_samples (int): The number of samples to generate.
        n_sweeps (int): The number of sweeps to perform for each sample.
        burn_in (int): The number of sweeps to discard as burn-in.

    Returns:
        list: A list of samples generated during the Metropolis sampling process.

    """
    current_state = initial_state
    samples = []

    for _ in range(n_samples):
        for _ in range(n_sweeps + burn_in):
            new_state = rbm.gibbs_sampling(current_state)

            acceptance_prob = min(1, np.exp(-local_energy(new_state, interact) +
                                            local_energy(current_state, interact)))
            if np.random.rand() < acceptance_prob:
                current_state = new_state

            if _ >= burn_in:
                samples.append(current_state)

    return samples


def importance_sampling(rbm, initial_state, interact, n_samples, n_sweeps, burn_in):
    """
    Performs importance sampling using a Restricted Boltzmann Machine (RBM).

    Args:
        rbm (RBM): An instance of the Restricted Boltzmann Machine class.
        initial_state (ndarray): An array representing the initial state of the system.
        interact (bool): A flag indicating whether to include interaction energy or not.
        n_samples (int): The number of samples to generate.
        n_sweeps (int): The number of sweeps to perform for each sample.
        burn_in (int): The number of sweeps to discard as burn-in.

    Returns:
        list: A list of samples generated during the importance sampling process.

    """
    current_state = initial_state
    samples = []

    for _ in range(n_samples):
        for _ in range(n_sweeps + burn_in):
            new_state = rbm.importance_sampling(current_state)

            acceptance_prob = min(1, np.exp(-local_energy(new_state, interact) +
                                            local_energy(current_state, interact)))

            if np.random.rand() < acceptance_prob:
                current_state = new_state

            if _ >= burn_in:
                samples.append(current_state)

    return samples


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

def check_convergence(energy_history, threshold=1e-6):
    """
    Checks if the energy values in the history have converged.

    Args:
        energy_history (list): A list of energy values.
        threshold (float): The threshold for convergence.

    Returns:
        bool: True if the energy values have converged, False otherwise.

    """
    if len(energy_history) < 2:
        return False

    energy_diff = np.abs(energy_history[-1] - energy_history[-2])
    return energy_diff < threshold


def save_constants(replace, converger, pcd=None):
    """
    Saves the constants used in the simulation to a file.

    Args:
        replace (bool): A flag indicating whether to replace the existing file.
        converger (str): The name of the converger used.
        pcd (tuple): A tuple containing the persistent chain and the persistent chain rate.

    Returns:
        dict: A dictionary containing the saved constants.

    """
    if not converger=='PCD':
        constants = {
            'N_SAMPLES': N_SAMPLES,
            'N_CONVERGE': N_CONVERGE,
            'N_HIDDEN': N_HIDDEN,
            'N_VISIBLE': N_VISIBLE,
            'LEARNING_RATE': LEARNING_RATE,
            'N_CHAINS': N_CHAINS,
            'N_SWEEPS': N_SWEEPS,
            'N_SCALE': N_SCALE,
            'omega': omega
        }
    else:
        constants = {
            'N_SAMPLES': N_SAMPLES,
            'N_CONVERGE': N_CONVERGE,
            'N_HIDDEN': N_HIDDEN,
            'N_VISIBLE': N_VISIBLE,
            'LEARNING_RATE': LEARNING_RATE,
            'N_CHAINS': N_CHAINS,
            'N_SWEEPS': N_SWEEPS,
            'N_SCALE': N_SCALE,
            'omega': omega,
            'persistent_chain': pcd[0],
            'PERSISTENT_CHAIN_RATE': pcd[1]
            }

    if replace:
        with open('../cached/constants_%s.txt' %converger, 'w') as f:
            for key, value in constants.items():
                f.write(f'{key} = {value}\n')
    
    return constants

def main(sample='metropolis', converger='PCD', interact=False, save=True, replace=True):
    
    if converger == 'PCD':
       persistent_chain = np.random.binomial(1, 0.5, size=(N_CHAINS, N_VISIBLE))
       PERSISTENT_CHAIN_RATE = 0.1
       constants = save_constants(False, converger, (persistent_chain, PERSISTENT_CHAIN_RATE))
    else:
        constants = save_constants(False, converger)
    t0 = time.time()
    rbm = None
    if not replace:
        try:
            with open('../cached/cached_RBM_%s' %converger, 'rb') as f:
                rbm = pickle.load(f)
                initial_state = rbm.state
                energy_history = rbm.energyList
                print('Found cached RBM (%s) model. If you want a new one, set replace=True' %converger)
        except FileNotFoundError:
            print("No cached RBM (%s) found. Generating a new RBM (%s)..." %converger)
            replace = True
    
    if replace:
        rbm = RBM(N_VISIBLE, N_HIDDEN, sample)
        initial_state = np.random.randn(N_SCALE, N_VISIBLE)
    
        energy_history = []
        convergence_flag = False
            
        for i in range(N_CONVERGE):
            if converger == 'CD':
                rbm.contrastive_divergence(N_CHAINS, LEARNING_RATE, initial_state)
            elif converger == 'PCD':
                rbm.persistent_contrastive_divergence(N_CHAINS, LEARNING_RATE, initial_state, persistent_chain, PERSISTENT_CHAIN_RATE)
            else:
                rbm.gradient_descent(LEARNING_RATE, initial_state)
                
            current_energy = local_energy(initial_state, interact)
            energy_history.append(current_energy)
    
            if check_convergence(energy_history):
                convergence_flag = True
                break
    
            initial_state = rbm.visible_probabilities(rbm.sample_hidden(initial_state))
            
        if convergence_flag:
            print('Training converged (%s). Converged at iteration %i after %.2fs\n' %(converger, i+1,time.time() - t0))
            rbm.state = initial_state
            rbm.energyList = energy_history
            iterations_used = i+1
            
            if save:
                    if converger=='PCD':
                        save_constants(replace, converger, (persistent_chain, PERSISTENT_CHAIN_RATE))
                    else:
                        save_constants(replace, converger)
                    with open('../cached/cached_RBM_%s' %converger, 'wb') as f:
                        pickle.dump(rbm, f)
                        print('RBM (%s) object saved to file.' %converger)
    
                    with open('../cached/constants_%s.txt' %converger, 'w') as f:
                        for key, value in constants.items():
                            f.write(f'{key} = {value}\n')
        else:
            iterations_used = i+1
            plt.plot(np.array(energy_history)/N_SCALE)
            plt.plot([analytical_energy(interact, 2, 2)]*iterations_used)
            plt.legend(['Numerical Energy', 'Analytical Energy'])
            plt.xlabel('Iteration')
            plt.ylabel('Energy (a.u.)')
            plt.title('Energy History (%s training)' %converger)
            plt.show()
            raise ValueError ('Training did not converge, aborting!')

    if sample == 'metropolis':
        num_processes = mp.cpu_count()
        pool = mp.Pool(processes=num_processes)
        chunk_size = N_SAMPLES // num_processes
        results = []

        for _ in range(num_processes):
            results.append(pool.apply_async(metropolis_sampling, args=(rbm, initial_state, interact, chunk_size,
                                                                       N_SWEEPS, N_SWEEPS // 5)))

        pool.close()
        pool.join()

        samples = []
        for result in results:
            samples.extend(result.get())
    elif sample == 'importance':
        num_processes = mp.cpu_count()
        pool = mp.Pool(processes=num_processes)
        chunk_size = N_SAMPLES // num_processes
        results = []

        for _ in range(num_processes):
            results.append(pool.apply_async(importance_sampling, args=(rbm, initial_state, interact, chunk_size,
                                                                       N_SWEEPS, N_SWEEPS // 5)))

        pool.close()
        pool.join()

        samples = []
        for result in results:
            samples.extend(result.get())
    else:
        raise ValueError("Invalid sample type selected.")


    energies = [local_energy(sample, interact) for sample in samples]
    blocking_results = blocking_analysis(np.array(energies)/N_SCALE)
    analytical_result = analytical_energy(interact, 2, 2)
    
    print("=================\nResults for:\nSamples: %i Hidden: %i Visible: %i Learning Rate: %.4f Sweep: %i Chain: %i\nWith %s sampling\n=================" %(chunk_size*num_processes, N_HIDDEN,
                                                                                                                                                     N_VISIBLE, LEARNING_RATE,
                                                                                                                                                     N_SWEEPS, N_CHAINS,
                                                                                                                                                     sample))
    print('Blocking analysis: %.2f +- %g' % (blocking_results[0], blocking_results[1]))
    print('Analytical result: %.2f' % analytical_result)
    
    plt.plot(np.array(energy_history)/N_SCALE)
    plt.plot([analytical_energy(interact, 2, 2)]*iterations_used)
    plt.legend(['Numerical Energy', 'Analytical Energy'])
    plt.xlabel('Iteration')
    plt.ylabel('Energy (a.u.)')
    plt.title('Energy History (%s training)' %converger)
    plt.show()


if __name__ == '__main__':
    main()
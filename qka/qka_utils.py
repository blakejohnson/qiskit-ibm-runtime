import numpy as np
from numpy.random import RandomState


def SPSA_parameters():
    """Return array of precomputed SPSA parameters.

    Returns:
        SPSA_params (array): [a, c, alpha, gamma, A]

        The i-th optimization step, i>=0, the parameters evolve as

            a_i = a / (i + 1 + A) ** alpha,
            c_i = c / (i + 1) ** gamma,

        for fixed coefficents a, c, alpha, gamma, A.

        Default Qiskit values are:
        SPSA_params = [2*np.pi*0.1, 0.1, 0.602, 0.101, 0]

    """
    SPSA_params = np.zeros((5))
    SPSA_params[0] = 0.01              # a
    SPSA_params[1] = 0.1               # c
    SPSA_params[2] = 0.602             # alpha  (alpha range [0.5 - 1.0])
    SPSA_params[3] = 0.101             # gamma  (gamma range [0.0 - 0.5])
    SPSA_params[4] = 0                 # A

    return SPSA_params


def spsa_step_one(lambdas, spsa_params, count):
    """Evaluate +/- perturbations of kernel parameters (theta):
    theta_+ = theta + c_spsa * delta and theta_- = theta - c_spsa * delta

    Args:
        lambdas (array): kernel parameters at step 'count' in SPSA optimization loop
        spsa_params (array): SPSA parameters
        count (int): the current step in the SPSA optimization loop.
    Returns:
        theta_plus (array): kernel parameters perturbed in + direction
        theta_minus (array): kernel parameters perturbed in - direction
        delta (array): random perturbation vector with elements {-1,1}

    """
    prng = RandomState(count)  # use randomstate to ensure repeatable Deltas

    c_spsa = float(spsa_params[1])/np.power(count+1, spsa_params[3])
    delta = 2*prng.randint(0, 2, size=np.shape(lambdas)[0]) - 1 # array of random integers in 2*[0, 2)-1 --> {-1,1}

    lambda_plus = lambdas + c_spsa * delta
    lambda_minus = lambdas - c_spsa * delta

    return lambda_plus, lambda_minus, delta


def spsa_step_two(cost_plus, cost_minus, lambdas, spsa_params, delta, count):
    """Evaluate one iteration of SPSA on SVM objective function F and
    return updated kernel parameters.

    F(alpha, lambda) = 1^T * alpha - (1/2) * alpha^T * Y * K * Y * alpha

    Args:
        cost_plus (float): Objective function F(alpha_+, lambda_+)
        cost_minus (float): Objective function F(alpha_-, lambda_-)
        lambdas (array): kernel parameters at step 'count' in SPSA optimization loop
        spsa_params (array): SPSA parameters
        delta (array): random perturbation vector with elements {-1,1}
        count(int): the current step in the SPSA optimization loop

    Returns:
        cost_final (list): estimate of updated SVM objective function F using average of F(alpha_+, lambda_+) and F(alpha_-, lambda_-) in format [[cost_final]]
        lambdas_new (array): updated values of the kernel parameters after one SPSA optimization step

    """
    a_spsa = float(spsa_params[0])/np.power(count+1+spsa_params[4], spsa_params[2])
    c_spsa = float(spsa_params[1])/np.power(count+1, spsa_params[3])

    g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa) # Approximate the gradient of SVM objective (note: 1/delta = delta)

    lambdas_new = lambdas - a_spsa * g_spsa # update kernel params from initial values to new values using estimate of gradient
    lambdas_new = lambdas_new.flatten()

    # Since direct value of alpha is not available because we did gradient ascent over alpha_+ and alpha_-, we cannot evaluate exact cost.
    # Take the average of cost_plus and cost_minus as an approximation:
    cost_final = (cost_plus+cost_minus)/2

    return cost_final, lambdas_new

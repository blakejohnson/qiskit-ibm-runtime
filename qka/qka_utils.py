import numpy as np
from numpy.random import RandomState


def SPSA_parameters():
    """Return array of precomputed SPSA parameters.

    Returns:
        SPSA_params (numpy.ndarray): [a, c, alpha, gamma, A]

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
    """Evaluate +/- perturbations of kernel parameters (lambdas).

    Args:
        lambdas (numpy.ndarray): kernel parameters at step 'count' in SPSA optimization loop
        spsa_params (numpy.ndarray): SPSA parameters
        count (int): the current step in the SPSA optimization loop
    Returns:
        lambda_plus (numpy.ndarray): kernel parameters in + direction
        lambda_minus (numpy.ndarray): kernel parameters in - direction
        delta (numpy.ndarray): random vector with elements {-1,1}
    """

    prng = RandomState(count)

    c_spsa = float(spsa_params[1])/np.power(count+1, spsa_params[3])
    delta = 2*prng.randint(0, 2, size=np.shape(lambdas)[0]) - 1

    lambda_plus = lambdas + c_spsa * delta
    lambda_minus = lambdas - c_spsa * delta

    return lambda_plus, lambda_minus, delta


def spsa_step_two(cost_plus, cost_minus, lambdas, spsa_params, delta, count):
    """Evaluate one iteration of SPSA on SVM objective function F and
    return updated kernel parameters.

    F(alpha, lambda) = 1^T * alpha - (1/2) * alpha^T * Y * K * Y * alpha

    Args:
        cost_plus (float): objective function F(alpha_+, lambda_+)
        cost_minus (float): objective function F(alpha_-, lambda_-)
        lambdas (numpy.ndarray): kernel parameters at step 'count' in SPSA optimization loop
        spsa_params (numpy.ndarray): SPSA parameters
        delta (numpy.ndarray): random vector with elements {-1,1}
        count(int): the current step in the SPSA optimization loop

    Returns:
        cost_final (float): estimate of updated SVM objective function F using average of F(alpha_+, lambda_+) and F(alpha_-, lambda_-)
        lambdas_new (numpy.ndarray): updated values of the kernel parameters after one SPSA optimization step
    """

    a_spsa = float(spsa_params[0])/np.power(count+1+spsa_params[4], spsa_params[2])
    c_spsa = float(spsa_params[1])/np.power(count+1, spsa_params[3])

    g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)

    lambdas_new = lambdas - a_spsa * g_spsa
    lambdas_new = lambdas_new.flatten()

    cost_final = (cost_plus + cost_minus) / 2

    return cost_final, lambdas_new

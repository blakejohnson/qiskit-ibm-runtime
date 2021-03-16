import numpy as np
from numpy.random import RandomState
from kernel_matrix import KernelMatrix
from cvxopt import matrix, solvers


class QKA:
    """The quantum kernel alignment algorithm."""

    def __init__(self, feature_map, backend, verbose=True):
        """Constructor.

        Args:
            feature_map (partial obj): the quantum feature map object
            backend (Backend): the backend instance
            verbose (bool): print output during course of algorithm
        """

        self.feature_map = feature_map
        self.feature_map_circuit = self.feature_map.construct_circuit # the feature map circuit not yet evaluated with input arguments
        self.backend = backend
        self.num_qubits = self.feature_map._num_qubits
        self.depth = self.feature_map._depth
        self.entangler_map = self.feature_map._entangler_map
        self.num_parameters = self.feature_map._num_parameters  # number of parameters (lambdas) in the feature map

        self.verbose = verbose
        self.result = {}

        self.kernel_matrix = KernelMatrix(feature_map=self.feature_map, backend=self.backend)



    def SPSA_parameters(self):
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



    def cvxopt_solver(self, K, y, C, max_iters=10000, show_progress=False):
        """Convex optimization of SVM objective using cvxopt.

        Args:
            K (numpy.ndarray): nxn kernel (Gram) matrix
            y (numpy.ndarray): nx1 vector of labels +/-1
            C (float): soft-margin penalty
            max_iters (int): maximum iterations for the solver
            show_progress (bool): print progress of solver

        Returns;
            ret (dict): results from the solver
        """

        if y.ndim == 1:
            y = y[:, np.newaxis]
        H = np.outer(y, y) * K
        f = -np.ones(y.shape)

        n = K.shape[1] # number of training points

        y = y.astype('float')

        P = matrix(H)                                                 # (nxn) yy^T * K, element-wise multiplication
        q = matrix(f)                                                 # (nx1) -ones
        G = matrix(np.vstack((-np.eye((n)), np.eye((n)))))            # for soft-margin, (2nxn) matrix with -identity (+identity) in top (bottom) half
        h = matrix(np.vstack((np.zeros((n,1)), np.ones((n,1)) * C)))  # for soft-margin, (2nx1) vector with 0's (C's) in top (bottom) half
        A = matrix(y, y.T.shape)                                      # (1xn) y
        b = matrix(np.zeros(1), (1, 1))                               # (1x1) zero


        solvers.options['maxiters'] = max_iters
        solvers.options['show_progress'] = show_progress

        ret = solvers.qp(P, q, G, h, A, b, kktsolver='ldl')

        return ret



    def spsa_step_one(self, lambdas, spsa_params, count):
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




    def spsa_step_two(self, cost_plus, cost_minus, lambdas, spsa_params, delta, count):
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




    def align_kernel(self, data, labels, lambda_initial=None, spsa_steps=10, C=1):
        """Align the quantum kernel.

        Uses SPSA for minimization over kernel parameters (lambdas) and
        convex optimization for maximization over lagrange multipliers (alpha):

        min_lambda max_alpha 1^T * alpha - (1/2) * alpha^T * Y * K_lambda * Y * alpha

        Args:
            data (numpy.ndarray): NxD array of training data, where N is the number of samples and D is the feature dimension
            labels (numpy.ndarray): Nx1 array of +/-1 labels of the N training samples
            lambda_initial (numpy.ndarray): Initial parameters of the quantum feature map
            spsa_steps (int): number of SPSA optimization steps
            C (float): penalty parameter for the soft-margin support vector machine

        Returns:
            result (dict): the results of kernel alignment
        """

        if lambda_initial is not None:
            lambdas = lambda_initial
        else:
            lambdas = np.random.uniform(-1.0, 1.0, size=(self.num_parameters))

        # Pre-computed spsa parameters:
        spsa_params = self.SPSA_parameters()

        # Save data at each SPSA run in the following lists:
        lambda_save = []       # updated kernel parameters after each spsa step
        cost_final_save = []   # avgerage cost at each spsa step
        cost_plus_save = []    # (+) cost at each spsa step
        cost_minus_save = []   # (-) cost at each spsa step
        program_data = []


        # #####################
        # Start the alignment:

        for count in range(spsa_steps):

            if self.verbose: print('\n\n  SPSA step {} of {}:\n'.format(count+1, spsa_steps))

            # (STEP 1 OF PSEUDOCODE)
            # First stage of SPSA optimization.

            lambda_plus, lambda_minus, delta = self.spsa_step_one(lambdas=lambdas, spsa_params=spsa_params, count=count)

            # (STEP 2 OF PSEUDOCODE)
            # Evaluate kernel matrix for the (+) and (-) kernel parameters.

            kernel_plus = self.kernel_matrix.construct_kernel_matrix(x1_vec=data, x2_vec=data, parameters=lambda_plus)
            kernel_minus = self.kernel_matrix.construct_kernel_matrix(x1_vec=data, x2_vec=data, parameters=lambda_minus)

            # (STEP 3 OF PSEUDOCODE)
            # Maximize SVM objective function over
            # support vectors in the (+) and (-) directions.

            ret_plus = self.cvxopt_solver(K=kernel_plus, y=labels, C=C)
            cost_plus = -1 * ret_plus['primal objective']

            ret_minus = self.cvxopt_solver(K=kernel_minus, y=labels, C=C)
            cost_minus = -1 * ret_minus['primal objective']

            # (STEP 4 OF PSEUDOCODE)
            # Second stage of SPSA optimization:
            # (one iteration of SPSA on SVM objective function F
            #  and return updated kernel parameters).

            cost_final, lambda_best = self.spsa_step_two(cost_plus=cost_plus, cost_minus=cost_minus,
                                                         lambdas=lambdas, spsa_params=spsa_params, delta=delta, count=count)

            if self.verbose: print('\n\n\033[92m Cost: {}\033[00m'.format(cost_final))

            lambdas = lambda_best # updated kernel parameters

            lambda_save.append(lambdas)
            cost_plus_save.append(cost_plus)
            cost_minus_save.append(cost_minus)
            cost_final_save.append(cost_final)

            program_data.append(self.kernel_matrix.results)


        # Evaluate aligned kernel matrix with optimized set of parameters averaged over last 10 SPSA steps:
        lambdas = np.sum(np.array(lambda_save)[-10:, :],axis = 0)/10
        kernel_best = self.kernel_matrix.construct_kernel_matrix(x1_vec=data, x2_vec=data, parameters=lambdas)

        self.result['aligned_kernel_parameters'] = lambdas
        self.result['aligned_kernel_matrix'] = kernel_best

        self.result['all_kernel_parameters'] = lambda_save
        self.result['all_final_cost_evaluations'] = cost_final_save
        self.result['all_positive_cost_evaluations'] = cost_plus_save
        self.result['all_negative_cost_evaluations'] = cost_minus_save

        self.result['program_data'] = program_data

        return self.result

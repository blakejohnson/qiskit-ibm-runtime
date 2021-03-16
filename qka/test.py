import numpy as np
from qiskit import Aer
from featuremaps import FeatureMapForrelation
from kernel_matrix import KernelMatrix
from qka import QKA
from sklearn import metrics
from sklearn.svm import SVC


num_samples=5  # number of samples per class in the input data
num_features=2 # number of features in the input data
depth=2        # depth of the feature map circuit
C=1            # SVM soft-margin penalty
spsa_steps=11  # number of SPSA iterations


bk = Aer.get_backend('qasm_simulator')


# create random test data and labels:

x_train = np.random.rand(2*num_samples, num_features)
y_train = np.concatenate((-1*np.ones(num_samples), np.ones(num_samples)))


# Define the feature map and its initial parameters:

fm = FeatureMapForrelation(feature_dimension=num_features, depth=depth, entangler_map=None)
lambda_initial = np.random.uniform(-1,1, size=(fm._num_parameters))



# Align the quantum kernel:

qka = QKA(feature_map=fm, backend=bk)
qka_results = qka.align_kernel(data=x_train, labels=y_train,
                               lambda_initial=lambda_initial,
                               spsa_steps=spsa_steps, C=C)


# Test the aligned kernel on test data:

x_test = np.random.rand(2*num_samples, num_features)
y_test = np.concatenate((-1*np.ones(num_samples), np.ones(num_samples)))

kernel_aligned = qka_results['aligned_kernel_matrix']
model = SVC(C=C, kernel='precomputed')
model.fit(X=kernel_aligned, y=y_train)

km = KernelMatrix(feature_map=fm, backend=bk)
kernel_test = km.construct_kernel_matrix(x1_vec=x_test, x2_vec=x_train, parameters=qka_results['aligned_kernel_parameters'])
labels_test = model.predict(X=kernel_test)
accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=labels_test)

print('Aligned Kernel | Balanced Test Accuracy: {}'.format(accuracy_test))

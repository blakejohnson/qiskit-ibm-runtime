import numpy as np
from qiskit import Aer
from featuremaps import FeatureMap
from kernel_matrix import KernelMatrix
from qka import QKA
from sklearn import metrics
from sklearn.svm import SVC

from oct2py import octave
octave.addpath('/Users/jen/Q/code/quantum_kernel_data')



# save data to csv:
# samples = 64 # per label
# features=2*7
# state=42 # setting the state for the random number generator
#
# data_plus, data_minus = octave.generate_data(samples, state, nout=2)
#
# dat = np.empty((samples*2, features))
# dat[::2,:] = data_plus.T
# dat[1::2,:] = data_minus.T
#
# lab = np.reshape(np.asarray([1,-1]*samples), (samples*2,1))
# full = np.concatenate((dat, lab), axis=1)
#
# np.savetxt("/Users/jen/Q/code/quantum_kernel_data/dataset_graph7.csv", full, delimiter=",")
# print('done')

# load data from csv:
import pandas as pd
df = pd.read_csv('/Users/jen/Q/code/quantum_kernel_data/dataset_graph7.csv',sep = ',', header = None)
dat = df.values

num_features = np.shape(dat)[1]-1 # feature dimension determined by dataset
num_train = 10
num_test = 10
C = 1
maxiters = 11

entangler_map=[[0,2],[3,4],[2,5],[1,4],[2,3],[4,6]]
initial_layout=None # [10,11,12,13,14,15,16]

x_train = dat[:2*num_train, :-1]
y_train = dat[:2*num_train, -1]

x_test = dat[2*num_train:2*(num_train+num_test), :-1]
y_test = dat[2*num_train:2*(num_train+num_test), -1]





# configure settings for the problem graph:
# num_features=2*7  # number of features in the input data
# num_train=10      # number of training samples per class
# num_test=10       # number of test samples per class
# C=1              # SVM soft-margin penalty
# maxiters=11       # number of SPSA iterations
#
# entangler_map=[[0,2],[3,4],[2,5],[1,4],[2,3],[4,6]]
# initial_layout=[10,11,12,13,14,15,16]
#
# # entangler_map=[[0,1],[1,2],[1,3]]
# # initial_layout=[0,1,2,4]
#
# # entangler_map=[[0,1],[2,3],[4,5],[6,7],[8,9],[1,2],[3,4],[5,6],[7,8]]
# # initial_layout = [9, 8, 11, 14, 16, 19, 22, 25, 24, 23]
#
#
# # Generate the data:
# state=42 # setting the state for the random number generator
# data_plus, data_minus = octave.generate_data(num_train+num_test, state, nout=2)
#
# x_train = np.concatenate((data_plus.T[:num_train], data_minus.T[:num_train]))
# y_train = np.concatenate((-1*np.ones(num_train), np.ones(num_train)))
#
# x_test = np.concatenate((data_plus.T[num_train:], data_minus.T[num_train:]))
# y_test = np.concatenate((-1*np.ones(num_test), np.ones(num_test)))






# Specify the backend
bk = Aer.get_backend('qasm_simulator')

# Define the feature map and its initial parameters:
initial_kernel_parameters = [0.1] # np.pi/2 should yield 100% test accuracy
fm = FeatureMap(feature_dimension=num_features, entangler_map=entangler_map)
km = KernelMatrix(feature_map=fm, backend=bk, initial_layout=initial_layout)

# Train and test the initial kernel:
kernel_init_train = km.construct_kernel_matrix(x1_vec=x_train, x2_vec=x_train, parameters=initial_kernel_parameters)
model = SVC(C=C, kernel='precomputed')
model.fit(X=kernel_init_train, y=y_train)

kernel_init_test = km.construct_kernel_matrix(x1_vec=x_test, x2_vec=x_train, parameters=initial_kernel_parameters)
labels_test = model.predict(X=kernel_init_test)
accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=labels_test)

print('Initial Kernel | Balanced Test Accuracy: {}'.format(accuracy_test))

# Align the parametrized kernel:
qka = QKA(feature_map=fm, backend=bk, initial_layout=initial_layout)
qka_results = qka.align_kernel(data=x_train, labels=y_train,
                               initial_kernel_parameters=initial_kernel_parameters,
                               maxiters=maxiters, C=C)

# Train and test the aligned kernel:
kernel_aligned = qka_results['aligned_kernel_matrix']
model = SVC(C=C, kernel='precomputed')
model.fit(X=kernel_aligned, y=y_train)

kernel_test = km.construct_kernel_matrix(x1_vec=x_test, x2_vec=x_train, parameters=qka_results['aligned_kernel_parameters'])
labels_test = model.predict(X=kernel_test)
accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=labels_test)

print('Aligned Kernel | Balanced Test Accuracy: {}'.format(accuracy_test))

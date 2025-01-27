# K-Nearest Neighbours Class - Jack Parry-Wingfield

# STEP 1 - Basic Import Statements

import numpy as np
import pandas as pd
#the output of plotting commands is displayed inline within frontends
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace         #for debugging

#it is important to set the seed for reproducibility as it initializes the random number generator
np.random.seed(1234)

# STEP 2 - Loading the Datasets

# penguin_data =
# (N, D, C) = ()

# heart_data =
# (N, D, C) = ()

# STEP 3 - Instantiating the I/O Data Structures (i.e. Input matrix and output vector)

# x_penguin, y_penguin =
# x_heart, y_heart =

# Splitting the data into test and train:

x_penguin_test, y_penguin_test = (pd.dataframe(), pd.dataframe())
x_penguin_train, y_penguin_train = (pd.dataframe(), pd.dataframe())

x_heart_test, y_heart_test = (pd.dataframe(), pd.dataframe())
x_heart_train, y_heart_train = (pd.dataframe(), pd.dataframe())

# STEP 4 - Initiation of the KNN Class, using the OOP paradigm specified in the assignment description

#euclidean =
#manhattan =

class KNN:

    # Constructor
    def __init__(self, K, distance_func):
        self.K = K
        self.distance_func = distance_func
        return

    # STEP 5 - Defining the 'fit' function - trains the model by merely remembering the dataset.
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.C = y.max() + 1 # C = the number of classes
        return self

    # STEP 6 - Defining the 'predict' function
    def predict(self, x_test): #pass in either x_penguin_test or x_heart_test
        """
        Makes a prediction using the stored training data and the test data given as an argument.
        """
        num_test = x_test.shape[0]

        # Calculate distances between training and test samples
        distances = self.distance_func(self.x.values[None, :, :],
                                 x_test.values[:, None, :])  # Convert to NumPy for compatibility

        # Create DataFrames for k-nearest neighbors indices and probability distribution
        knns = pd.DataFrame(index=range(num_test), columns=range(self.K), dtype=int) # k-nearest neighbours
        y_prob = pd.DataFrame(index=range(num_test), columns=range(self.C), dtype=float).fillna(0) # probability distributions

        for i in range(num_test):
            # Get indices of K closest training samples for the i-th test sample
            knns.iloc[i, :] = distances[i].argsort()[:self.K]  # Get the indices of the K smallest distances

            # Count occurrences of each class in the K nearest neighbors
            neighbor_labels = self.y.iloc[knns.iloc[i, :]].values  # Retrieve the class labels of the K neighbors
            class_counts = pd.Series(neighbor_labels).value_counts().reindex(range(self.C), fill_value=0)

            # Store class probabilities for the i-th test sample
            y_prob.iloc[i, :] = class_counts / self.K

        return y_prob, knns

    def evaluate_acc(self, y_true, y_pred):
        accuracy = (y_pred == y_true).mean()
        print(f'accuracy is {accuracy * 100:.1f}.')


# STEP 7 - fitting the model for penguin

model_penguin = KNN(K=3) # modify K as required

y_prob_penguin, knns_penguin = model_penguin.fit(x_penguin_train, y_penguin_train).predict(x_penguin_test)
print('knns_penguin shape:', knns_penguin.shape)
print('y_prob_penguin shape:', y_prob_penguin.shape)

#To get hard predictions by choosing the class with the maximum probability
y_pred_penguin = y_prob_penguin.idxmax(axis=1)
model_penguin.evaluate_acc(y_penguin_train, y_pred_penguin)

# Boolean series to later slice the indexes of correct and incorrect predictions
correct_penguin = y_penguin_test == y_pred_penguin
incorrect_penguin = ~correct_penguin  # The tilde (~) operator negates the boolean series


# STEP 8 - fitting the model for heart

model_heart = KNN(K=3) # modify K as required

y_prob_heart, knns_heart = model_heart.fit(x_heart_train, y_heart_train).predict(x_heart_test)
print('knns_heart shape:', knns_heart.shape)
print('y_prob_heart shape:', y_prob_heart.shape)

#To get hard predictions by choosing the class with the maximum probability
y_pred_heart = y_prob_heart.idxmax(axis=1)
model_heart.evaluate_acc(y_heart_train, y_pred_heart)

# Boolean series to later slice the indexes of correct and incorrect predictions
correct_heart = y_heart_test == y_pred_heart
incorrect_heart = ~correct_heart  # The tilde (~) operator negates the boolean series

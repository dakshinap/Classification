'''
Created on Sep 30, 2017

@author: dakshina
'''

import numpy as np

from keras.callbacks import Callback
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras import metrics

from sklearn.model_selection import StratifiedKFold





class CrossValidator:
    debug = False
    
    def __init__(self, Examples, Labels, n_folds=10, epochs=50):
        """CrossValidator(Examples, Labels, n_folds)
        Given a list of training examples in Examples and a corresponding
        set of class labels in Labels, train and evaluate a learner
        using cross validation.
 
        """
        
        # Set up an n-fold test and call train_and_evaluate_model
        # for each fold.  Keep statistics and report summary
        # information of mean and variance.
        
        # Randmize examples within each fold
        kfold = StratifiedKFold(n_folds, shuffle=True)
        one_hot_labels = np_utils.to_categorical(Labels)
        # Generate indices that can be used to split into training
        # and test data, e.g. examples[train_idx]
        accuracy = []
        for (train_idx, test_idx) in kfold.split(Examples, Labels):
        # normally, we would gather results about each fold
            accuracy.append(self.train_and_evaluate__model(Examples, one_hot_labels, train_idx, test_idx, 100, epochs))
        accuracy = np.array(accuracy)
        mu, sd = np.mean(accuracy), np.std(accuracy)
        print()
        print('mean accuracy {} and standard deviation {}'.format(mu,sd))

    def train_and_evaluate__model(self, examples, labels, train_idx, test_idx, 
                                  batch_size=100, epochs=100):
        """train_and_evaluate__model(examples, labels, train_idx, test_idx,
                batch_size=100)
                
        Given:
            examples - List of examples in column major order
                (# of rows is feature dim)
            labels - list of corresponding labels
            train_idx - list of indices of examples and labels to be learned
            test_idx - list of indices of examples and labels of which
                the system should be tested.
        Optional arguments
            batch_size - size of minibatch
            epochs - # of epochs to compute
            
        Returns error rate [0, 1]
        """
    
        # train and evaluate model
        model = Sequential()

        # Hidden Layers
        model.add(Dense(20, activation='relu', input_dim = 2000))
        model.add(Dense(20, activation='relu', input_dim = 20))

        # Output Layer
        model.add(Dense(11, activation='softmax', input_dim = 20))
        
        model.compile(optimizer = "Adam",
               loss = "categorical_crossentropy",
               metrics = [metrics.categorical_accuracy])

        model.summary()

        model.fit(examples[train_idx], labels[train_idx], batch_size, epochs)

        # Returns list of metrics
        results = model.evaluate(examples[test_idx], labels[test_idx])
        # model.metrics_names tells us what was measured
        # here: ['loss', 'categorical_accuracy’]

        return results[1] 	# accuracy
        # In some fields, it is common to report error: 1 - accuracy
        
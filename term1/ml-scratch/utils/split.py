#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:31:05 2019

@author: arimoto
"""

def train_test_split(X,y,test_size=0.25,
                     random_state=None,shuffle=True,stratify=None):
    """
    Split the data to be learned and tested.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
      data to be learned and tested
    y : ndarray, shape (n_samples, )
      objective labels
    test_size : float (0<test_size<1)(default: 0.25)
      set the rate of test size
    random_state : int
      set the pseudo-random number to be used in RandomStateGenerator
    shuffle : boolean (default:True)
      shuffle before split or not. If False, set stratify as None.
    stratify : array-like or None
      array for stratified sampling

    Returns
    ----------
    X_train : ndarray, shape (n_samples, n_features)
      data to be learned
    X_test : ndarray, shape (n_samples, n_features)
      data to be tested
    y_train : ndarray, shape (n_samples, )
      labels for X_train
    y_test : ndarray, shape (n_samples, )
      labels for X_test
    """
    import numpy as np
    # Error if feature samples number does not corresponds to y number.
    if X.shape[0] != y.shape[0]:
        raise ValueError("X samples number({}) is not same as y {}.".format(
                X.shape[0], y.shape[0]))
    
    # make several parameters to be used
    n_samples = X.shape[0]
    n_train = np.floor((1-test_size) * n_samples).astype(int)
    n_test = n_samples - n_train
    classes = np.unique(y)
    n_classes = len(classes)
    class_counts = np.bincount(y)
    class_indices = np.split(np.argsort(y, kind='mergesort'),
                             np.cumsum(class_counts)[:-1])
    
    # Case1: Shuffle=False and stratify=None
    if shuffle is False and stratify is None:
        X_test = X[:n_test]
        X_train = X[n_test:(n_test + n_train)]
        y_test = y[:n_test]
        y_train = y[n_test:(n_test + n_train)]
        
        return X_train, X_test, y_train, y_test
    
    # Case2: Shuffle=False and stratify=y
    elif shuffle is False and stratify is not None:
        raise ValueError("If 'shuffle' parameter is False, "
                         "then 'stratify' parameter should be None.")
    
    # Case3: Shuffle=True and stratify=None
    elif shuffle is True and stratify is None:
        rng = np.random.RandomState(seed=random_state)
        # shuffle and split
        permutation = rng.permutation(n_samples)
        ind_test = permutation[:n_test]
        ind_train = permutation[n_test:(n_test + n_train)]
        
        X_train = X[ind_train]
        X_test = X[ind_test]
        y_train = y[ind_train]
        y_test = y[ind_test]
        
        yield X_train
        yield X_test
        yield y_train
        yield y_test
    
    # Case4: Shuffle=True and stratify=y
    else:
        def extracting_func(class_counts, n_draws, rng):
            """
            Stratified sampling at random a certain number(n_draws) of samples 
            from population in class_counts.
            
            """
            # assign each number of samples to be extracted per each class
            continuous = n_draws * (class_counts / class_counts.sum())
            floored = np.floor(continuous)
            need_to_add = int(n_draws - floored.sum())
            # determine which classes should be added one more because of flooring
            if need_to_add > 0:
                remainder = continuous - floored
                # sort the remaining values in an unascending manner
                values = np.sort(np.unique(remainder))[::-1]
                for value in values:
                    inds, = np.where(remainder == value)
                    # set the number of value to be added
                    add_now = min(len(inds), need_to_add)
                    # determine at random where should be added
                    inds = rng.choice(inds, size=add_now, replace=False)
                    floored[inds] += 1
                    # repeat until when 'need to add' becomes 0
                    need_to_add -= add_now
                    if need_to_add == 0:
                        break
            return floored.astype(np.int)
        
        # set a number of samples to be selected per each class
        rng = np.random.RandomState(seed=random_state)
        n_i = extracting_func(class_counts, n_train, rng)
        class_counts_remaining = class_counts - n_i
        t_i = extracting_func(class_counts_remaining, n_test, rng)
        
        train = []
        test = []
        
        # select at random which indices should be assigned to train and test set
        for i in range(n_classes):
            permutation = rng.permutation(class_counts[i])
            perm_indices_class_i = class_indices[i].take(
                    permutation,mode='clip')
            train.extend(perm_indices_class_i[:n_i[i]])
            test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])
        
        ind_train = rng.permutation(train)
        ind_test = rng.permutation(test)
        
        X_train = X[ind_train]
        X_test = X[ind_test]
        y_train = y[ind_train]
        y_test = y[ind_test]
        
        yield X_train
        yield X_test
        yield y_train
        yield y_test

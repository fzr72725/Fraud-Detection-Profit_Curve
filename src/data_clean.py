import pandas as pd
import numpy as np

def check_nan(df, cols):
    '''Check and print specific columns in given dataframe for nan/null counts
    Inputs
    ----------
    df  : pandas DataFrame
    cols  : list of string
    Returns
    -------
    None
    '''
    print 'Dataframe total row count: ', df.shape[0]
    nan_cnt = 0
    for c in cols:
        if sum(df[c].isnull())>0:
            print c , ' has ' , sum(df[c].isnull()) , 'null/NAN values'
            nan_cnt += 1
    print 'There are {} columns with nan/null values'.format(nan_cnt)
    return None

def check_zero(df, col):
    '''Check and print numeric columns in given dataframe for 0 counts
    Inputs
    ----------
    df  : pandas DataFrame
    col  : numeric column name
    Returns
    -------
    None
    '''
    if sum(df[col]==0)>0:
        print 'Column', col,':',sum(df[col]==0),'rows are 0'
    else:
        print 'No 0 values in column {}'.format(col)
        
def undersample(X, y, tp=0.5):
    '''Randomly discards negative observations from X & y to achieve the
    target proportion of positive to negative observations.
    Inputs
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : default 0.5, float - range [0.5, 1], target proportion of positive class observations
    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    '''
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    negative_sample_rate = (pos_count * (1 - tp)) / (neg_count * tp)
    negative_keepers = np.random.choice(a=[False, True], size=neg_count,
                                        p=[1 - negative_sample_rate,
                                           negative_sample_rate])
    X_negative_undersampled = X_neg[negative_keepers]
    y_negative_undersampled = y_neg[negative_keepers]
    X_undersampled = np.vstack((X_negative_undersampled, X_pos))
    y_undersampled = np.concatenate((y_negative_undersampled, y_pos))

    return X_undersampled, y_undersampled

def oversample(X, y, tp=0.5):
    """Randomly choose positive observations from X & y, with replacement
    to achieve the target proportion of positive to negative observations.
    Inputs
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : default 0.5, float - range [0, 1], target proportion of positive class observations
    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    positive_range = np.arange(pos_count)
    positive_size = (tp * neg_count) / (1 - tp)
    positive_idxs = np.random.choice(a=positive_range,
                                     size=int(positive_size),
                                     replace=True)
    X_positive_oversampled = X_pos[positive_idxs]
    y_positive_oversampled = y_pos[positive_idxs]
    X_oversampled = np.vstack((X_positive_oversampled, X_neg))
    y_oversampled = np.concatenate((y_positive_oversampled, y_neg))

    return X_oversampled, y_oversampled


def div_count_pos_neg(X, y):
    """Helper function to divide X & y into positive and negative classes
    and counts the number in each.
    Inputs
    ----------
    X : ndarray - 2D
    y : ndarray - 1D
    Returns
    -------
    negative_count : Int
    positive_count : Int
    X_positives    : ndarray - 2D
    X_negatives    : ndarray - 2D
    y_positives    : ndarray - 1D
    y_negatives    : ndarray - 1D
    """
    negatives = y == 0
    positives = y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives
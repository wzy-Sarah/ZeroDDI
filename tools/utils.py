import os
import os.path as osp
import random
import numpy as np
import torch
import csv
import sys
import logging

logger = logging.getLogger(__name__)

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        


try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def ddie_compute_metrics(task_name, preds, labels, every_type=False):
        label_list = ('Mechanism', 'Effect', 'Advise', 'Int.')
        p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average='micro')
        result = {
            "Precision": p,
            "Recall": r,
            "microF": f
        }
        if every_type:
            for i, label_type in enumerate(label_list):
                p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average='micro')
                result[label_type + ' Precision'] = p
                result[label_type + ' Recall'] = r
                result[label_type + ' F'] = f
        return result

    def pretraining_compute_metrics(task_name, preds, labels, every_type=False):
        acc = accuracy_score(y_pred=preds, y_true=labels)
        result = {
            "Accuracy": acc,
        }
        return result

def get_top_n(n, matrix):
    """Gets probability a number n and a matrix,
    returns a new matrix with largest n numbers in each row of the original matrix."""

    return (-matrix).argsort()[:, 0:n]


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def GetAccuracy(Y_Pred, Probabilities, Y_True, TrueClassIndices, eps=1e-15):
    """ Returns Accuracy when multi-label are provided for each instance. It will be counted true if predicted y is among the true labels
    Args:
        Y_Pred (int array): the predicted labels
        Probabilities (float [][] array): the probabilities predicted for each class for each instance
        Y_True (int[] array): the true labels, for each instance it should be a list
    """
    count_true = 0
    count_true_3 = 0
    count_true_5 = 0
    count_true_10 = 0
    top_10_classes = get_top_n(10, Probabilities)
    for i in range(len(Y_Pred)):
        if Y_Pred[i] == Y_True[i]:
            count_true += 1
        if len(intersection(top_10_classes[i], [TrueClassIndices[i]])) > 0:
            count_true_10 += 1
        if len(intersection(top_10_classes[i][:5], [TrueClassIndices[i]])) > 0:
            count_true_5 += 1
        if len(intersection(top_10_classes[i][:3], [TrueClassIndices[i]])) > 0:
            count_true_3 += 1

    Evaluations = {"Accuracy": (float(count_true) / len(Y_Pred)), 
                   "Top3Acc": (float(count_true_3) / len(Y_Pred)), "Top5Acc": (float(count_true_5) / len(Y_Pred)),
                   "Top10Acc": (float(count_true_10) / len(Y_Pred)), "Probabilities": Probabilities,
                   "Ypred": Y_Pred, "Ytrue": Y_True}
    return Evaluations
    
def softmax( X, theta=1.0, axis=None):
      """
      Compute the softmax of each element along an axis of X.

      Parameters
      ----------
      X: ND-Array. Probably should be floats.
      theta (optional): float parameter, used as a multiplier
          prior to exponentiation. Default = 1.0
      axis (optional): axis to compute values along. Default is the
          first non-singleton axis.

      Returns an array the same size as X. The result will sum to 1
      along the specified axis.
      """

      # make X at least 2d
      y = np.atleast_2d(X)

      # find axis
      if axis is None:
          axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

      # multiply y against the theta parameter,
      y = y * float(theta)

      # subtract the max for numerical stability
      y = y - np.expand_dims(np.max(y, axis=axis), axis)

      # exponentiate y
      y = np.exp(y)

      # take the sum along the specified axis
      ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

      # finally: divide elementwise
      p = y / ax_sum

      # flatten if X was 1D
      if len(X.shape) == 1: p = p.flatten()

      return p


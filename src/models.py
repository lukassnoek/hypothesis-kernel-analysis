import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import pairwise_kernels, pairwise_distances, roc_auc_score

# This file contains all AU names!
AU_SET = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()


def _softmax_2d(arr, beta):
    """ Vectorized softmax implementation including an inverse temperature
    parameter (beta). It assumes a 2D array with stimuli x features (similarities). """
    scaled = beta * arr
    num = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    denom = np.sum(num, axis=1, keepdims=True)
    return num / denom


class KernelClassifier(BaseEstimator, ClassifierMixin):  
    """ A "Kernel classifier" that computes the probability of an emotion
    being present in the face based on the associated theoretical action unit
    configuration. """
    def __init__(self, au_cfg, param_names, kernel='linear', ktype='similarity', binarize_X=False, beta=1, normalization='softmax', kernel_kwargs=None):
        """ Initializes an KernelClassifier object.
        
        au_cfg : dict
            Dictionary with theoretical kernels (e.g. `mappings['Darwin']` from mappings.py)
        param_names : list
            List with parameter names (e.g., ['AU1', 'AU2', etc.])
        kernel : str
            Name of kernel; anything that is accepted by sklearn's `pairwise_kernels` or `pairwise_distances`
        ktype : str
            Type of kernel; either 'similarity' (uses `pairwise_kernels`) or 'distance' (uses `pairwise_distances`)
        binarize_X : bool
            Whether to binarize the configuration (0 > = 1, else 0)
        normalization : str
            How to normalize the similarities ('softmax' or 'linear')
        beta : int/float
            Beta parameter for the softmax function. 
        kernel_kwargs : dict
            Optional extra arguments to be used in either `pairwise_distances` or `pairwise_similarity`
        """

        self.au_cfg = au_cfg
        self.kernel = kernel
        self.ktype = ktype
        self.binarize_X = binarize_X
        self.beta = beta
        self.param_names = param_names
        self.normalization = normalization
        self.kernel_kwargs = kernel_kwargs 
        self.Z_ = None
        self.labels_ = None
        self.cls_idx_ = None
        self.sim_ = None

    def set_params(self, **params):
        """ Sets params, but slightly differently than scikit-learn,
        by respecting the **kernel_kwargs argument. """
        for param, value in params.items():
            if param not in self.__dict__:
                if self.kernel_kwargs is None:
                    self.kernel_kwargs = dict()
                self.kernel_kwargs[param] = value
            else:
                setattr(self, param, value)

        return self

    def get_params(self, deep=True):
        """ Gets all parameters from object. """
        params = self.__dict__.copy()
        for p in self.__dict__.keys():
            if p[-1] == '_':  # exclude from init
                del params[p]

        return params

    def _setup(self):
        """ Sets up kernels by creating a vector per class. Only done once (i.e., 
        the first time calling `fit`). """
        
        # Labels are the different classes (e.g., happy, angry, fear, etc.)
        self.labels_ = np.array(list(self.au_cfg.keys()))

        cls_idx = []  # find how many configs each class has
        for i, (_, cfg) in enumerate(sorted(self.au_cfg.items())):
            if isinstance(cfg, dict):
                nr = len(cfg)
            else:
                nr = 1
            
            cls_idx.extend([i] * nr)

        self.cls_idx_ = np.array(cls_idx)

        # Initialize array Z, which holds the numerical config
        P = len(self.param_names)  # number of AUs
        self.Z_ = np.zeros((self.cls_idx_.size, P))  # theory configuration

        # Loop across different classes and their configuration
        # (e.g., happy: ['AU6', 'AU12'], sad: ['AU4'])
        i = 0
        for _, cfg in sorted(self.au_cfg.items()):
            # If cfg is a dict, then there are multiple configuations
            if isinstance(cfg, dict):
                # Theory vector is 2D: n_configs x n_params
                for combi in cfg.values():
                    for c in combi:
                        self.Z_[i, self.param_names.index(c)] = 1
                
                    i += 1
            else:  # Theory vector is a 1D vector (of shape P)
                for c in cfg:
                    self.Z_[i, self.param_names.index(c)] = 1
                i += 1

        self.Z_ = pd.DataFrame(self.Z_, columns=self.param_names,
                               index=self.labels_[self.cls_idx_])

    def fit(self, X=None, y=None):
        """ Doesn't fit any parameters but is included for scikit-learn
        compatibility. Also, the kernels are setup here, as it is apparently
        good practice to defer any computing to the fit() call. """
        if self.kernel_kwargs is None:
            self.kernel_kwargs = dict()
        
        if self.Z_ is None:
            # Only set up the theory vector once
            self._setup()

    def predict_proba(self, X):
        """ Predicts a probabilistic target label.
        
        Parameters
        ----------
        X : numpy array
            A 2D numpy array of shape N (samples) x P (features)
        y : numpy array
            A 1D numpy array of shape N (samples)
        """
        return self._predict(X)

    def predict(self, X):
        """ Predicts a discrete target label. First calls `predict`, which returns a
        probabilistic prediction and then takes the argmax.
        
        Parameters
        ----------
        X : numpy array
            A 2D numpy array of shape N (samples) x P (features)
        """
        # Get probabilistic predictions and take the argmax
        probs = self._predict(X)
        preds = probs.argmax(axis=1)
        #argmax_idx = probs == probs.max(axis=1, keepdims=True)
        #ties = argmax_idx.sum(axis=1) > 1
        #print(np.where(argmax_idx[ties])[1].shape)
        return preds

    def score(self, X, y):
        """ Computes the model performance score. Hardcoded to be `roc_auc_score`. """
        y = pd.get_dummies(y)
        preds = self.predict_proba(X)
        return roc_auc_score(y, preds, average='micro')

    def _predict(self, X, y=None):
        """ Predicts the probability of a sample belonging to each class
        based on the corresponding kernel. """

        if X.shape[1] != len(self.param_names):
            raise ValueError("X does not have the expected number of features.")
        
        if self.binarize_X:  # binarize input features
            X = (X > 0).astype(int)

        if self.ktype == 'similarity':
            sim = pairwise_kernels(X, self.Z_, metric=self.kernel, **self.kernel_kwargs)
        else:
            delta = pairwise_distances(X, self.Z_, metric=self.kernel, **self.kernel_kwargs)
            # Convert distance to similarity 
            sim = 1 / (1 + delta)

        # Set NaNs to 0 (i.e., undefined similarity equals 0)
        sim = np.nan_to_num(sim)

        # If there are multiple configurations per class, take the max!
        self.sim_ = np.hstack(
            [sim[:, i == self.cls_idx_].max(axis=1, keepdims=True)
             for i in np.unique(self.cls_idx_)]
        )

        # Set zeros to a small number (EPS) to make sure softmax doesn't crash        
        EPS = 1e-10
        self.sim_[self.sim_ == 0] = EPS
        if self.normalization == 'softmax':
            probs = _softmax_2d(self.sim_, self.beta)
        else:  # linear normalization
            probs = self.sim_ / self.sim_.sum(axis=1, keepdims=True)

        return probs

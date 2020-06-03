import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


AU_SET = np.loadtxt('data/au_names_new.txt', dtype=str).tolist()


def _softmax_2d(arr, beta):
    """ Vectorized softmax implementation with inverse temperature
    parameters (beta). """
    scaled = beta * arr
    num = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    denom = np.sum(num, axis=1, keepdims=True)
    return num / denom


class TheoryKernelClassifier(BaseEstimator, ClassifierMixin):  
    """ A "Theory-kernel classifier" that computes the probability of an emotion
    being present in the face based on the associated theoretical action unit
    configuration. """
    def __init__(self, kernel_dict, binarize_X=False, normalize=True, scale_dist=True, beta_sm=3):
        """ Initializes an TheoryKernelClassifier object.
        
        kernel_dict : dict
            Dictionary with theoretical kernels
        binarize_X : bool
            Whether to binarize the configuration (0 > = 1, else 0)
        normalize : bool
            Whether to normalize the distances with the sum of the AU-vector
        scale_dist : bool
            Whether to scale the distances from 0 to 1 (recommended)
        beta_sm : int/float
            Beta parameter for the softmax function. 
        """

        self.kernel_dict = kernel_dict
        self.binarize_X = binarize_X
        self.normalize = normalize
        self.scale_dist = scale_dist
        self.beta_sm = beta_sm
        self.params = AU_SET
        self.kernels = dict()  # 'filled' in _setup() call
        self.labels = ()  # same
    
    def _setup(self):
        """ Sets up kernels by creating a vector per class. """

        P = len(self.params)        
        for clss, cfg in self.kernel_dict.items():
            # If dict, then there are multiple options
            if isinstance(cfg, dict):  # Kernel is initially 2D: n_options x n_params
                kernel = np.zeros((len(cfg), P))
                for i, combi in cfg.items():
                    for c in combi:
                        print(self.params)
                        kernel[i, self.params.index(c)] = 1            
            else:  # Kernel is a 1D vector (of shape P)
                kernel = np.zeros(P)
                for c in cfg:
                    print(type(self.params))
                    kernel[self.params.index(c)] = 1
            
            self.labels += (clss,)
            self.kernels[clss] = kernel

    def fit(self, X=None, y=None):
        """ Doesn't fit any parameters but is included for scikit-learn
        compatibility. Also, the kernels are setup here, as it is apparently
        good practice to defer any computing to the fit() call. """

        self._setup()
    
    def predict_proba(self, X, y=None):
        """ Predicts a probabilistic target label.
        
        Parameters
        ----------
        X : numpy array
            A 2D numpy array of shape N (samples) x P (features)
        y : numpy array
            A 1D numpy array of shape N (samples)
        """
        return self._predict(X, y)

    def predict(self, X, y=None):
        """ Predicts a discrete target label.
        
        Parameters
        ----------
        X : numpy array
            A 2D numpy array of shape N (samples) x P (features)
        y : numpy array
            A 1D numpy array of shape N (samples)
        """
        probs = self._predict(X, y)
        ties = np.squeeze(probs == probs.max())
        if np.sum(ties) > 0:
            pred = np.random.choice(np.arange(ties.size)[ties])
        else:
            pred = np.argmax(probs)
            
        return pred
        
    def _predict(self, X, y=None):
        """ Predicts the probability of a sample belonging to each class
        based on the corresponding kernel. """

        if X.shape[1] != len(self.params):
            raise ValueError("X does not have the expected number of features.")
        
        if self.binarize_X:
            X = (X > 0).astype(int)

        N, P = X.shape
        K = len(self.labels)
        self.dist = np.zeros((N, K))
        for i, (_, kernel) in enumerate(self.kernels.items()):
            if kernel.ndim > 1:
                dist = np.zeros((K, N))
                for i in range(K):
                    # IDEA: use different distance metrics
                    tmp = (X - kernel[i, :]) ** 2
                    if self.normalize:
                        # Normalize by kernel sq sum
                        tmp /= np.sum(kernel[i, :])  # should I square this?
                    
                    # Compute euclidean distance
                    dist[i, :] = np.sqrt(np.sum(tmp, axis=1))

                # Take max (or mean?) of different options
                dist = np.max(dist,  axis=0)
            else:
                dist = (X - kernel) ** 2
                if self.normalize:
                    dist /= np.sum(kernel)  # should I square this?
                
                dist = np.sqrt(np.sum(dist, axis=1))
            
            self.dist[:, i] = dist
        
        if self.scale_dist:
            minim = self.dist.min(axis=1, keepdims=True)
            maxim = self.dist.max(axis=1)
            rnge =  maxim - np.squeeze(minim)
            self.dist = (self.dist - minim) / rnge[:, np.newaxis]
        
        EPS = 1e-10
        self.dist[self.dist == 0] = EPS
        probs = _softmax_2d(1 / self.dist, self.beta_sm)
        return probs
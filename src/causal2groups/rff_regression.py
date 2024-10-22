'''
Regression on top of random fourier features.
'''
import numpy as np
from jax import jit
import jax.numpy as jnp
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from autograd_minimize import minimize

class RBFRFFTransform:
    def __init__(self, input_dim:int, output_dim:int, seed:int):
        self.rng = np.random.default_rng(seed)
        self.output_dim = output_dim
        self.proj_mat = self.rng.standard_normal(size=(input_dim,output_dim))
        self.shift = self.rng.uniform(low=0, high=(2.*np.pi), size=output_dim)

    def transform(self, X:np.ndarray, bandwidth:float=1.0, include_identity:bool=False):
        X_proj = X @ self.proj_mat
        X_rff = np.sqrt(2/self.output_dim) * np.cos(X_proj/np.sqrt(bandwidth) + self.shift)

        if include_identity:
            return(np.hstack([np.ones((X.shape[0], 1)), X,X_rff]))
        else:
            return(X_rff)
    
    def multiscale_transform(self, X:np.ndarray, bandwidths:list[float], include_identity:bool=False):
        X_rff = np.hstack([ self.transform(X, bandwidth=bwidth, include_identity=((i==0) and include_identity)) for i, bwidth in enumerate(bandwidths)])
        return(X_rff)
    
    def __call__(self,  X:np.ndarray, bandwidth:float|list[float], include_identity:bool=False):
        if isinstance(bandwidth, float):
            bandwidth=[bandwidth]
        return(self.multiscale_transform(X=X, bandwidths=bandwidth, include_identity=include_identity))


class RFFRegression:
    def __init__(self, n_bandwidths, input_dim:int, rff_dims:int, include_identity:bool, seed:int):
        self.rng = np.random.default_rng(seed)
        self.n_bandwidths = n_bandwidths
        self.transform = RBFRFFTransform(input_dim=input_dim, output_dim=rff_dims, seed=seed)
        self.include_identity = include_identity

    def fit_via_cv(self, X:np.ndarray, y:np.ndarray, loss, n_cv:int=3, weights:np.ndarray=None, verbose:bool=False):
        '''
            Loss takes arguments (prediction, y, weights)
        '''

        n = X.shape[0]
        if weights is None:
            weights = np.ones(n)

        sq_dists = squareform(pdist(X, metric='sqeuclidean'))
        sorted_dists = np.sort(sq_dists, axis=1)
        med_dists = np.median(sorted_dists, axis=0)
        dmin = np.min(med_dists[med_dists>0])
        dmax = med_dists[-1]
        bandwidths = np.logspace(np.log10(dmin), 2*(np.log10(dmax) - np.log10(dmin)), num=self.n_bandwidths)

        ## Create CV splits
        idx_splits = np.array_split(self.rng.permutation(n), n_cv)
        pred_lookup = {}
        loss_lookup = {}
        for bwidth in tqdm(bandwidths, disable=(not verbose)):
            X_transform = self.transform(X=X, bandwidth=bwidth, include_identity=self.include_identity)
            for i, test_idx in enumerate(idx_splits):
                train_idx = np.setdiff1d(np.arange(n), test_idx)
                X_train = jnp.array(X_transform[train_idx], dtype=jnp.float32)
                y_train = jnp.array(y[train_idx], dtype=jnp.float32)
                weights_train = jnp.array(weights[train_idx], dtype=jnp.float32)

                train_loss = lambda w: loss(jnp.dot(X_train, w), y_train, weights_train)
                w0 = np.ones(X_train.shape[1], dtype=np.float32)
                res = minimize(jit(train_loss), w0, method='L-BFGS-B', backend='jax')
                w = jnp.array(res['x'], dtype=jnp.float32)

                X_test = jnp.array(X_transform[test_idx], dtype=jnp.float32)
                y_test = jnp.array(y[test_idx], dtype=jnp.float32)
                weights_test = jnp.array(weights[test_idx], dtype=jnp.float32)

                y_test_pred = jnp.dot(X_test, w)
                pred_lookup[bwidth, i] = np.array(y_test_pred)
                loss_lookup[bwidth, i] = loss(y_test_pred, y_test, weights_test).item()
        
        ## Choose best bandwidth
        losses = {bwidth:0.0 for bwidth in bandwidths}
        for bwidth, i in loss_lookup.keys():
            losses[bwidth] += loss_lookup[bwidth, i]
        
        bwidths = list(losses.keys())
        self.bandwidth = bwidths[np.argmin(list(losses.values()))]
        self.cv_preds = np.empty_like(y)
        for i, test_idx in enumerate(idx_splits):
            self.cv_preds[test_idx] = pred_lookup[self.bandwidth, i]

        ## Fit model on all data
        X_transform = self.transform(X=X, bandwidth=self.bandwidth, include_identity=self.include_identity)
        X_train = jnp.array(X_transform, dtype=jnp.float32)
        y_train = jnp.array(y, dtype=jnp.float32)
        weights_train = jnp.array(weights, dtype=jnp.float32)

        train_loss = lambda w: loss(jnp.dot(X_train, w), y_train, weights_train)
        w0 = np.ones(X_train.shape[1], dtype=np.float32)
        res = minimize(jit(train_loss), w0, method='L-BFGS-B', backend='jax')
        self.w = res['x']

    def predict(self, X_pred:np.ndarray):
        X_transform = self.transform(X=X_pred, bandwidth=self.bandwidth, include_identity=self.include_identity)
        pred = np.dot(X_transform, self.w)
        return(pred)
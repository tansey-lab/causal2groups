import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm import tqdm


def gcv(Lam, Qty, lam):
    '''
        Computes generalized cross-validation estimate (Golub et al., 1979)
        for kernel ridge regression with kernel matrix K, responses y, reg parameter lam.

        Inputs:
            Lam: eigenvalues of K
            Qty: Q.T @ y, where K = Q @ np.diag(Lam) @ Q.T is the eigendecomposition of K
            lam: regularization parameter
    '''

    mse = np.mean(Qty*(1 - 2*Lam/(Lam + lam) + np.square(Lam/(Lam + lam)))*Qty)
    mean_tr_H = (1./lam)*np.mean( Lam - np.square(Lam)/(Lam + lam))
    return(mse/np.square(1-mean_tr_H))

def loo_predictions(K_diag, Q, Lam, y, lam):
    '''
        Computes leave-one-out predictions for kernel ridge regression 
        with kernel matrix K, responses y, regularization parameter lam.

        Inputs:
            K: kernel matrix
            Q, Lam: eigendecomposition K = Q @ np.diag(Lam) @ Q.T 
            y: responses
            lam: regularization parameter
    '''

    y_full_pred = np.linalg.multi_dot([Q, np.diag(Lam/(Lam+lam)), Q.T, y])
    V = np.dot(np.diag(Lam/np.sqrt((Lam+lam))), Q.T)
    beta = (1./lam)*(K_diag - np.sum(np.square(V), axis=0))

    y_part_pred = (y_full_pred - (beta*y))/(1-beta)
    return(y_part_pred)


def loo_residuals(K_diag, Q, Lam, y, lam):
    '''
        Computes leave-one-out residiuals for kernel ridge regression 
        with kernel matrix K, responses y, regularization parameter lam.

        Inputs:
            K: kernel matrix
            Q, Lam: eigendecomposition K = Q @ np.diag(Lam) @ Q.T 
            y: responses
            lam: regularization parameter
    '''

    y_part_pred = loo_predictions(K_diag, Q, Lam, y, lam)

    return(y - y_part_pred)

class KernelRidgeRegression:
    def __init__(self, bandwidth_neighbors, reg_params):
        super().__init__()
        if isinstance(bandwidth_neighbors, int):
            bandwidth_neighbors = [bandwidth_neighbors]

        if isinstance(bandwidth_neighbors, float):
            reg_params = [reg_params]

        self.bandwidth_neighbors = bandwidth_neighbors
        self.reg_params = reg_params
        self.eigen_lookup = None
        self.kdiag_lookup = None

    ## Fit via generalized cross-validation
    def fit_via_gcv(self, 
                    X:np.ndarray, 
                    y:np.ndarray, 
                    keep_eigen_lookup:bool=False, 
                    recalculate_eigen_lookup:bool=True, 
                    verbose:bool=False):
        
        self.X = X
        self.y = y

        recalc_flag = recalculate_eigen_lookup or (self.eigen_lookup is None)

        if recalc_flag:
            n_points, _ = X.shape
            bandwidth_neighbors = np.unique(np.maximum(np.minimum(self.bandwidth_neighbors, (n_points-1)),1))

            self.sq_dists = squareform(pdist(X, metric='sqeuclidean'))

            ## Candidate bandwidths are given by median distance to nearest neighbors
            sorted_dists = np.sort(self.sq_dists, axis=1)
            med_dists = np.median(sorted_dists, axis=0)
            bandwidths = med_dists[bandwidth_neighbors]

            ## Double the number of bandwidths just to be sure
            bandwidths = np.concatenate([bandwidths, np.max(bandwidths)*np.logspace(np.log10(2), 2, num=bandwidths.shape[0])])
        else:
            bandwidths = list(self.eigen_lookup.keys())
            
        self.cv_errs = {}
        eigen_lookup = {}
        kdiag_lookup = {}
        for bwidth in tqdm(bandwidths, disable=(not verbose)):
            if recalc_flag:
                K = np.exp(-0.5*self.sq_dists/bwidth)
                Lam, Q = np.linalg.eigh(K)
                eigen_lookup[bwidth] = Lam, Q
                kdiag_lookup[bwidth] = np.diag(K)
            else:
                Lam, Q = self.eigen_lookup[bwidth]
                eigen_lookup[bwidth] = self.eigen_lookup[bwidth]
                kdiag_lookup[bwidth] = self.kdiag_lookup[bwidth]

            Qty = np.dot(Q.T, y)
            errs = [gcv(Lam, Qty, lam) for lam in self.reg_params]
            self.cv_errs.update({(bwidth, lam):err for lam,err in zip(self.reg_params, errs)})

        keys = list(self.cv_errs.keys())
        vals = list(self.cv_errs.values())
        self.bandwidth, self.lam = keys[np.argmin(vals)]

        ## Get final model
        self.Lam, self.Q = eigen_lookup[self.bandwidth]
        self.K_diag = kdiag_lookup[self.bandwidth]

        if keep_eigen_lookup:
            self.eigen_lookup = eigen_lookup
            self.kdiag_lookup = kdiag_lookup

    def loo_predictions(self):
        return(loo_predictions(self.K_diag, self.Q, self.Lam, self.y, self.lam))

    def loo_residuals(self):
        return(loo_residuals(self.K_diag, self.Q, self.Lam, self.y, self.lam))


    def fit(self, X, y, bandwidth, lam):
        self.X = X
        self.y = y
        self.sq_dists = squareform(pdist(X, metric="sqeuclidean"))
        self.bandwidth = bandwidth
        self.lam = lam 

        K = np.exp(-0.5*self.sq_dists/self.bandwidth)
        self.Lam, self.Q = np.linalg.eigh(K)
        self.K_diag = np.diag(K)

    def update(self, X=None, y=None, bandwidth=None, lam=None):
        update_K_inv = False

        if X is not None:
            self.X = X
            self.sq_dists = squareform(pdist(X, metric="sqeuclidean"))
            update_K_inv = True

        if y is not None:
            self.y = y

        if lam is not None:
            self.lam = lam
            update_K_inv = True
        
        if bandwidth is not None:
            self.bandwidth = bandwidth
            update_K_inv = True
        
        if update_K_inv:
            K = np.exp(-0.5*self.sq_dists/self.bandwidth)
            self.Lam, self.Q = np.linalg.eigh(K)
            self.K_diag = np.diag(K)


    def predict(self, X_pred):
        sq_dists_cross = cdist(X_pred, self.X, metric="sqeuclidean") ## n_pred x n

        K_cross = np.exp(-0.5*sq_dists_cross/self.bandwidth) 

        y_pred = np.linalg.multi_dot([K_cross, self.Q, np.diag(1/(self.Lam + self.lam)), self.Q.T, self.y])

        return(y_pred)

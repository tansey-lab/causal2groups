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

class KernelRidgeRegression:
    def __init__(self, n_bandwidths, reg_params):
        super().__init__()

        if isinstance(reg_params, float):
            reg_params = [reg_params]

        self.n_bandwidths = n_bandwidths
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
        self.X_stdv = np.std(X, axis=0, keepdims=True)
        self.X_mean = np.mean(X, axis=0, keepdims=True)
        self.X_transform = (X- self.X_mean)/self.X_stdv

        self.y = y
        self.y_stdv = np.std(y)
        self.y_mean = np.mean(y)
        self.y_transform = (y- self.y_mean)/self.y_stdv

        recalc_flag = recalculate_eigen_lookup or (self.eigen_lookup is None)

        if recalc_flag:
            n_points, _ = X.shape

            self.sq_dists = squareform(pdist(self.X_transform, metric='sqeuclidean'))

            ## Candidate bandwidths are given by median distance to nearest neighbors
            sorted_dists = np.sort(self.sq_dists, axis=1)
            med_dists = np.median(sorted_dists, axis=0)
            dmin = np.min(med_dists[med_dists>0])
            dmax = med_dists[-1]
            bandwidths = np.logspace(np.log10(dmin), 2*(np.log10(dmax) - np.log10(dmin)), num=self.n_bandwidths)
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

            Qty = np.dot(Q.T, self.y_transform)
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
        y_pred_transform = loo_predictions(self.K_diag, self.Q, self.Lam, self.y_transform, self.lam)
        y_pred = y_pred_transform*self.y_stdv + self.y_mean
        return(y_pred)

    def loo_residuals(self):
        y_pred = self.loo_predictions()
        return(self.y-y_pred)


    def predict(self, X_pred):
        X_pred_transform = (X_pred- self.X_mean)/self.X_stdv

        sq_dists_cross = cdist(X_pred_transform, self.X_transform, metric="sqeuclidean") ## n_pred x n

        K_cross = np.exp(-0.5*sq_dists_cross/self.bandwidth) 

        y_pred_transform = np.linalg.multi_dot([K_cross, self.Q, np.diag(1/(self.Lam + self.lam)), self.Q.T, self.y_transform])

        y_pred = y_pred_transform*self.y_stdv + self.y_mean

        return(y_pred)

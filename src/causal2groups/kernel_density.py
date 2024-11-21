import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm, trange
from itertools import product
from scipy.stats import norm

class ConditionalKDE:
    def __init__(self):
        super().__init__()

    ## Fit via leave-one-out cross-validation
    def fit_via_loo_cv(self, X:np.ndarray, y:np.ndarray, n_neighbors, bandwidth_neighbor_fracs, verbose:bool=False):   
        self.X = X
        self.y = y
        n_points, dim = X.shape

        if isinstance(n_neighbors, int):
            n_neighbors = [n_neighbors]

        if isinstance(bandwidth_neighbor_fracs, float):
            bandwidth_neighbor_fracs = [bandwidth_neighbor_fracs]
        
        bandwidth_neighbor_fracs = np.array(bandwidth_neighbor_fracs)
        bandwidth_neighbors = (bandwidth_neighbor_fracs*n_points).astype(int)
        bandwidth_neighbors = np.unique(np.maximum(np.minimum(bandwidth_neighbors, (n_points-1)),2))
        
        n_neighbors = np.unique(np.minimum(n_neighbors, (n_points - 1)))
        max_neighbors = np.max(np.concatenate([n_neighbors, bandwidth_neighbors]))+1

        tree_x = KDTree(X)
        D_x, inds = tree_x.query(X, k=max_neighbors,sort_results=True)
        D_x = D_x[:,1:]
        inds = inds[:,1:]
        
        tree_y = KDTree(y[:,np.newaxis])
        D_y,_ = tree_y.query(y[:,np.newaxis], k=max_neighbors, sort_results=True)
        
        ## x-bandwidths
        hxs = np.median(D_x[:,bandwidth_neighbors-1], axis=0)
        hxs = hxs[hxs>0]

        ## y-bandwidths
        hys = np.median(D_y[:,bandwidth_neighbors-1], axis=0)
        hys = hys[hys>0]
        hys = np.union1d(hxs, hys)

        ## Redefine y distance in terms of x distance
        D_y = np.abs(y[:,np.newaxis] - y[inds])
        
        params = list(product(n_neighbors, hxs, hys))
        log_likehoods = []
        for n_neighbors, hx, hy in tqdm(params, disable=(not verbose)):
            sq_x_dists = np.square(D_x[:, :n_neighbors])
            sq_y_dists = np.square(D_y[:, :n_neighbors])

            hx_sq = np.square(hx)
            k_x = np.power(1.0/(2*np.pi*hx_sq), 0.5*dim)*np.exp(-sq_x_dists/(2*hx_sq))

            hy_sq = np.square(hy)
            k_y = np.power(1.0/(2*np.pi*hy_sq), 0.5)*np.exp(-sq_y_dists/(2*hy_sq))

            numerators = np.mean(k_x*k_y, axis=-1)
            denominators = np.mean(k_x, axis=-1)
            

            vals = numerators/denominators
            with np.errstate(divide = 'ignore'):
                curr_log_likelihood = np.sum(np.log(vals))
            log_likehoods.append(curr_log_likelihood)

        ## Choose most likely
        best_i = np.argmax(log_likehoods)
        self.n_neighbors, self.hx, self.hy = params[best_i]

    def fit(self, X, y, n_neighbors, hx, hy):
        self.X = X
        self.y = y
        self.n_neighbors = n_neighbors
        self.hx = hx 
        self.hy = hy
        
    
    def predict_density(self, X_pred, y_pred, exclude_zero_dists:bool=True, verbose:bool=False):
        n_pred_x, dim = X_pred.shape
        if len(y_pred.shape)==2:
            assert y_pred.shape[0] in [1, n_pred_x], "if len(y_pred.shape)==2, then must have y_pred.shape[0]==X_pred.shape[0] or 1"
        elif len(y_pred.shape)>2:
            raise ValueError("len(y_pred.shape) > 2")
        elif y_pred.shape[0] != n_pred_x:
            y_pred = y_pred[np.newaxis,:]
        else:
            y_pred = y_pred[:,np.newaxis]

        n_pred_y = y_pred.shape[1]
        
        tree = KDTree(self.X)
        D_x, inds = tree.query(X_pred, k=self.n_neighbors, sort_results=False)
        mask = D_x!=0 if exclude_zero_dists else np.ones_like(D_x, dtype=bool)
        y_train_neighbors = self.y[inds] ## n_pred_x x n_neighbors

        sq_x_dists = np.square(D_x)
        hx_sq = np.square(self.hx)
        k_x = np.power(1.0/(2*np.pi*hx_sq), 0.5*dim)*np.exp(-sq_x_dists/(2*hx_sq)) ## n_pred_x x n_neighbors
        k_x = k_x * mask ## Mask out restricted points
        denominators = np.mean(k_x, axis=-1) ## n_pred_x
        denominators = np.abs(denominators)

        ## Run over the grid of values to predict for y
        chunk_size = 10
        chunks = np.array_split(np.arange(n_pred_y), max(n_pred_y//chunk_size, 1))

        results = []
        for chunk in tqdm(chunks, disable=(not verbose)):
            y_chunk = y_pred[:,chunk]
            D_y =  np.abs(y_train_neighbors[:,:,np.newaxis] - y_chunk[:,np.newaxis,:]) ## n_pred_x x n_neighbors x n_chunk
            sq_y_dists = np.square(D_y)

            hy_sq = np.square(self.hy)
            k_y = np.power(1.0/(2*np.pi*hy_sq), 0.5)*np.exp(-sq_y_dists/(2*hy_sq)) ## n_pred_x x n_neighbors x n_chunk
            
            numerators = np.mean(k_x[:,:,np.newaxis]*k_y, axis=1) ## n_pred_x x n_chunk
            
            vals = numerators/denominators[:,np.newaxis] ## n_pred_x x n_chunk
            results.append(vals)
        vals = np.concatenate(results, axis=-1)
        return(vals)
    
    def bootstrap(self, X_pred:np.ndarray, y_pred:np.ndarray, n:int=100, verbose:bool=False):
        ## Copy out the data
        X_train = self.X.copy()
        y_train = self.y.copy()
        
        results = []
        n_train = X_train.shape[0]
        for _ in trange(n, disable=(not verbose)):
            bootstrap_idx = np.random.choice(n_train, size=n_train, replace=True) 
            self.fit(X=X_train[bootstrap_idx], 
                     y=y_train[bootstrap_idx], 
                     hx=self.hx, 
                     hy=self.hy, 
                     n_neighbors=self.n_neighbors)
            results.append(self.predict_density(X_pred, y_pred, verbose=False))
        
        ## Reset the data
        self.fit(X=X_train, 
                 y=y_train,
                 hx=self.hx,
                 hy=self.hy,
                 n_neighbors=self.n_neighbors)
        return(np.stack(results))
    

    def predict_cdf(self, X_pred, y_pred, exclude_zero_dists:bool=True, verbose:bool=False):
        _, dim = X_pred.shape
        tree = KDTree(self.X)
        D_x, inds = tree.query(X_pred, k=self.n_neighbors, sort_results=False)
        mask = D_x!=0 if exclude_zero_dists else np.ones_like(D_x, dtype=bool)
        y_train_neighbors = self.y[inds] ## n_pred_x x n_neighbors

        sq_x_dists = np.square(D_x)
        hx_sq = np.square(self.hx)
        k_x = np.power(1.0/(2*np.pi*hx_sq), 0.5*dim)*np.exp(-sq_x_dists/(2*hx_sq)) ## n_pred_x x n_neighbors
        k_x = k_x * mask ## Mask out restricted points
        denominators = np.mean(k_x, axis=-1) ## n_pred_x
        denominators = np.abs(denominators)

        cdfs = norm.cdf(y_pred[:,np.newaxis], loc=y_train_neighbors, scale=self.hy) ## n_pred_x x n_neighbors
        numerators = np.mean(k_x*cdfs, axis=1) ## n_pred_x

        vals = numerators/denominators ## n_pred_x
        return(vals)
    
    def predict_mean(self, X_pred, exclude_zero_dists:bool=True):
        _, dim = X_pred.shape
        tree = KDTree(self.X)
        D_x, inds = tree.query(X_pred, k=self.n_neighbors, sort_results=False)
        mask = D_x!=0 if exclude_zero_dists else np.ones_like(D_x, dtype=bool)
        y_train_neighbors = self.y[inds] ## n_pred_x x n_neighbors

        sq_x_dists = np.square(D_x)
        hx_sq = np.square(self.hx)
        k_x = np.power(1.0/(2*np.pi*hx_sq), 0.5*dim)*np.exp(-sq_x_dists/(2*hx_sq)) ## n_pred_x x n_neighbors
        k_x = k_x * mask ## Mask out restricted points
        denominators = np.mean(k_x, axis=-1) ## n_pred_x
        denominators = np.abs(denominators)

        means = y_train_neighbors ## n_pred_x x n_neighbors
        numerators = np.mean(k_x*means, axis=1) ## n_pred_x

        vals = numerators/denominators ## n_pred_x
        return(vals)
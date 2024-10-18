import numpy as np
from causal2groups.kernel_density import ConditionalKDE
from scipy.stats import false_discovery_control
from tqdm import trange

class KernelFrequentist:
    def __init__(self, 
                 kernel_n_neighbors:list, 
                 kernel_bandwidth_neighbors:list,
                 n_bootstraps:int=100,
                 density_thresh:float=0.05,
                 n_grid:int=75,
                 verbose:bool=False):

        self.null_model = ConditionalKDE()
        self.kernel_n_neighbors = kernel_n_neighbors
        self.kernel_bandwidth_neighbors = kernel_bandwidth_neighbors
        self.n_bootstraps = n_bootstraps
        self.density_thresh = density_thresh
        self.n_grid = n_grid
        self.verbose = verbose

    def fit(self, X, Y, T):
        '''Fits a nonadditive causal two-groups model and performs selection on
        the treated population with control of the FDR at the target level.'''

        if self.verbose:
            print('Fitting null model.')

        self.null_model.fit_via_loo_cv(X=X[T==0], y=Y[T==0], n_neighbors=self.kernel_n_neighbors, bandwidth_neighbors=self.kernel_bandwidth_neighbors)

        if self.verbose:
            print('Drawing bootstrap samples from treatment model.')

        self.grid = np.linspace(np.min(Y), np.max(Y), num=self.n_grid)
        null_grid_boot = self.null_model.bootstrap(X, self.grid)

        null_grid_median = np.median(null_grid_boot, 0.5, axis=0)
        ## Linearly interpolate densities at grid
        null_density = []
        for i in trange(null_grid_median.shape[0]):
            null_density.append(np.interp(Y[i], self.grid, null_grid_median[i]))
        self.null_density = np.array(null_density)

    def calculate_fdr(self, T:np.ndarray, H:np.ndarray, fdr_levels:np.ndarray):
        p_vals = self.null_density.copy()
        p_vals_treated = p_vals[T==1]
        H_treated = H[T==1]
        qvals = false_discovery_control(p_vals_treated)
        n_pos = np.sum(H_treated)
        n_pos = np.sum(H_treated)

        fdr_observed = np.zeros_like(fdr_levels)
        power_observed = np.zeros_like(fdr_levels)
        for i, alpha in enumerate(fdr_levels):
            mask = qvals<=alpha
            num_sel = np.sum(mask)
            num_neg = num_sel - np.sum(H_treated[mask])
            fdr_observed[i] = 0 if num_sel==0 else num_neg/num_sel
            power_observed[i] = np.sum(H_treated[mask])/n_pos

        return(fdr_observed, power_observed)
import numpy as np
from tqdm import trange
from causal2groups.kernel_density import ConditionalKDE
from causal2groups.kernel_ridge import KernelRidgeRegression
from causal2groups.utils import posterior_selection_fdr, posterior_selection, posterior_selection_empirical_control


class KernelNonparametricCausal2G:
    def __init__(self, 
                 kernel_n_neighbors:list, 
                 kernel_bandwidth_neighbor_fracs:list,
                 n_bootstraps:int=100,
                 bootstrap_quantile:float=0.2,
                 empirical_control:bool=True,
                 density_thresh:float=0.05,
                 n_grid:int=75,
                 verbose:bool=False):

        self.null_model = ConditionalKDE()
        self.treatment_model = ConditionalKDE()
        self.kernel_n_neighbors = kernel_n_neighbors
        self.kernel_bandwidth_neighbor_fracs = kernel_bandwidth_neighbor_fracs
        self.n_bootstraps = n_bootstraps
        self.bootstrap_quantile = bootstrap_quantile
        self.empirical_control = empirical_control
        self.density_thresh = density_thresh
        self.n_grid = n_grid
        self.verbose = verbose
        self.null_mean_model = None
        self.treat_mean_model = None


    def fit(self, X:np.ndarray, Y:np.ndarray, T:np.ndarray):
        '''Fits a nonadditive causal two-groups model and performs selection on
        the treated population with control of the FDR at the target level.'''
        self.X = X.copy()
        self.Y = Y.copy()
        self.T = T.copy()

        if self.verbose:
            print('Fitting null model.')

        self.null_model.fit_via_loo_cv(X=X[T==0], 
                                       y=Y[T==0], 
                                       n_neighbors=self.kernel_n_neighbors, 
                                       bandwidth_neighbor_fracs=self.kernel_bandwidth_neighbor_fracs, 
                                       verbose=self.verbose)

        if self.verbose:
            print("Null model bandwidths:", self.null_model.hx, self.null_model.hy)
            print('Drawing bootstrap samples from null model.')

        self.grid = np.linspace(np.min(Y), np.max(Y), num=self.n_grid)
        null_grid_boot = self.null_model.bootstrap(X, self.grid, verbose=self.verbose)
        null_grid_boot = np.clip(null_grid_boot, a_min=1e-100, a_max=1e100)

        if self.verbose:
            print('Fitting treatment model.')

        self.treatment_model.fit_via_loo_cv(X=X[T==1], 
                                            y=Y[T==1], 
                                            n_neighbors=self.kernel_n_neighbors, 
                                            bandwidth_neighbor_fracs=self.kernel_bandwidth_neighbor_fracs, 
                                            verbose=self.verbose)

        if self.verbose:
            print("Treatment model bandwidths:", self.treatment_model.hx, self.treatment_model.hy)
            print('Drawing bootstrap samples from treatment model.')

        treat_grid_boot = self.treatment_model.bootstrap(X, self.grid, verbose=self.verbose)
        treat_grid_boot = np.clip(treat_grid_boot, a_min=1e-100, a_max=1e100)

        ## Take quantiles across bootstrap samples
        self.treat_grid_upper = np.quantile(treat_grid_boot, 1-self.bootstrap_quantile, axis=0)
        self.treat_grid_lower = np.quantile(treat_grid_boot, self.bootstrap_quantile, axis=0)

        self.null_grid_upper = np.quantile(null_grid_boot, 1-self.bootstrap_quantile, axis=0)
        self.null_grid_lower = np.quantile(null_grid_boot, self.bootstrap_quantile, axis=0)

        ## Estimate conservative prior at each data point.
        self.pi_star = self.estimate_conservative_prior(null_grid_lower=self.null_grid_lower, 
                                                        treat_grid_upper=self.treat_grid_upper, 
                                                        Y=Y)
        
        if self.verbose:
            print('Estimating null posterior')
        
        ## Estimate conservative prior at each data point.
        self.null_posterior = self.estimate_null_posterior(null_grid_upper=self.null_grid_upper, 
                                                           treat_grid_lower=self.treat_grid_lower,
                                                           pi_star=self.pi_star, Y=Y)


    def estimate_conservative_prior(self, 
                                    null_grid_lower:np.ndarray,
                                    treat_grid_upper:np.ndarray, 
                                    Y:np.ndarray):
        N = Y.shape[0]

        pi_star = np.zeros(N)
        fracs_conservative = treat_grid_upper/null_grid_lower
        for i in trange(N):
            density_thresh = self.density_thresh
            mask = treat_grid_upper[i]>self.density_thresh
            while not np.any(mask):
                density_thresh = density_thresh*0.9
                mask = treat_grid_upper[i]>density_thresh
            pi_star[i] = np.clip(1 - np.quantile(fracs_conservative[i,mask], 0.01), 0.0, 1.0)

        return(pi_star)

    def estimate_null_posterior(self, 
                                null_grid_upper:np.ndarray, 
                                treat_grid_lower:np.ndarray, 
                                pi_star:np.ndarray, 
                                Y:np.ndarray):
        N = Y.shape[0]

        ## Linearly interpolate densities at grid
        treat_density_lower = []
        null_density_upper = []

        for i in trange(N):
            treat_density_lower.append(np.interp(Y[i], self.grid, treat_grid_lower[i]))
            null_density_upper.append(np.interp(Y[i], self.grid, null_grid_upper[i]))

        treat_density_lower = np.array(treat_density_lower)
        null_density_upper = np.array(null_density_upper)


        null_posterior = (1.-pi_star)*null_density_upper/treat_density_lower
        null_posterior = np.clip(null_posterior, 0.0, 1.0)
        return(null_posterior)

    def predict_ite(self):
        ## Fit a model to the means
        if self.null_mean_model is None:
            self.null_mean_model = KernelRidgeRegression(n_bandwidths=6, reg_params=np.logspace(-5, 2, num=50))
            self.null_mean_model.fit_via_gcv(X=self.X[self.T==0], y=self.Y[self.T==0], verbose=self.verbose)

        if self.treat_mean_model is None:
            self.treat_mean_model = KernelRidgeRegression(n_bandwidths=6, reg_params=np.logspace(-5, 2, num=50))
            self.treat_mean_model.fit_via_gcv(X=self.X[self.T==1], y=self.Y[self.T==1], verbose=self.verbose)


        self.null_preds = np.empty_like(self.Y, dtype=float)
        self.null_preds[self.T==0] = self.null_mean_model.loo_predictions()
        self.null_preds[self.T==1] = self.null_mean_model.predict(X_pred=self.X[self.T==1])


        self.treat_preds = np.empty_like(self.Y, dtype=float)
        self.treat_preds[self.T==1] = self.treat_mean_model.loo_predictions()
        self.treat_preds[self.T==0] = self.treat_mean_model.predict(X_pred=self.X[self.T==0])


        pi_star = np.clip(self.pi_star, a_min=0.01, a_max=0.99)
        self.alt_preds = (1./pi_star)*(self.treat_preds - (1 - pi_star)*self.null_preds)
        ite_upper = self.alt_preds - self.null_preds
        ite_lower = self.treat_preds - self.null_preds

        return(ite_upper, ite_lower)

    def select(self, fdr_levels:np.ndarray, empirical_control:bool=False):
        if empirical_control:
            selections = posterior_selection_empirical_control(self.null_posterior.copy(), self.T, fdr_levels)
        else:
            selections = posterior_selection(self.null_posterior.copy(), self.T, fdr_levels)
        return(selections)
    
    def calculate_fdr(self, T:np.ndarray, H:np.ndarray, fdr_levels:np.ndarray, empirical_control:bool=False):
        null_posterior = self.null_posterior.copy()

        fdr_observed, power_observed = posterior_selection_fdr(null_posterior, T, H, fdr_levels, empirical_control=empirical_control)

        return(fdr_observed, power_observed)
    

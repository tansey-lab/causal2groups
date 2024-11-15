import numpy as np
from tqdm import trange
from causal2groups.kernel_density import ConditionalKDE
from causal2groups.kernel_ridge import KernelRidgeRegression

class KernelNonadditiveCausal2G:
    def __init__(self, 
                 kernel_n_neighbors:list, 
                 kernel_bandwidth_neighbors:list,
                 n_bootstraps:int=100,
                 bootstrap_quantile:float=0.2,
                 empirical_control:bool=True,
                 density_thresh:float=0.05,
                 n_grid:int=75,
                 verbose:bool=False):

        self.null_model = ConditionalKDE()
        self.treatment_model = ConditionalKDE()
        self.kernel_n_neighbors = kernel_n_neighbors
        self.kernel_bandwidth_neighbors = kernel_bandwidth_neighbors
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
                                       bandwidth_neighbors=self.kernel_bandwidth_neighbors, 
                                       verbose=self.verbose)

        if self.verbose:
            print('Drawing bootstrap samples from null model.')

        self.grid = np.linspace(np.min(Y), np.max(Y), num=self.n_grid)
        null_grid_boot = self.null_model.bootstrap(X, self.grid)

        if self.verbose:
            print('Fitting treatment model.')

        self.treatment_model.fit_via_loo_cv(X=X[T==1], 
                                            y=Y[T==1], 
                                            n_neighbors=self.kernel_n_neighbors, 
                                            bandwidth_neighbors=self.kernel_bandwidth_neighbors, 
                                            verbose=self.verbose)

        if self.verbose:
            print('Drawing bootstrap samples from treatment model.')

        treat_grid_boot = self.treatment_model.bootstrap(X, self.grid, verbose=self.verbose)

        ## Take quantiles across bootstrap samples
        treat_grid_upper = np.quantile(treat_grid_boot, 1-self.bootstrap_quantile, axis=0)
        treat_grid_lower = np.quantile(treat_grid_boot, self.bootstrap_quantile, axis=0)

        null_grid_upper = np.quantile(null_grid_boot, 1-self.bootstrap_quantile, axis=0)
        null_grid_lower = np.quantile(null_grid_boot, self.bootstrap_quantile, axis=0)
        

        ## Estimate conservative prior at each data point.
        self.pi_star = self.estimate_conservative_prior(null_grid_lower=null_grid_lower, 
                                                        treat_grid_upper=treat_grid_upper, 
                                                        Y=Y)
        
        if self.verbose:
            print('Estimating null posterior')
        
        ## Estimate conservative prior at each data point.
        self.null_posterior = self.estimate_null_posterior(null_grid_upper=null_grid_upper, 
                                                           treat_grid_lower=treat_grid_lower,
                                                           pi_star=self.pi_star, Y=Y)


        

    def estimate_conservative_prior(self, 
                                    null_grid_lower:np.ndarray,
                                    treat_grid_upper:np.ndarray, 
                                    Y:np.ndarray):
        N = Y.shape[0]

        pi_star = np.zeros(N)
        fracs_conservative = treat_grid_upper/null_grid_lower
        for i in trange(N):
            mask = treat_grid_upper[i]>self.density_thresh
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
            self.null_mean_model = KernelRidgeRegression(n_bandwidths=6, reg_params=np.logspace(-5, 5, num=50))
            self.null_mean_model.fit_via_gcv(X=self.X[self.T==0], y=self.Y[self.T==0])

        if self.treat_mean_model is None:
            self.treat_mean_model = KernelRidgeRegression(n_bandwidths=6, reg_params=np.logspace(-5, 5, num=50))
            self.treat_mean_model.fit_via_gcv(X=self.X[self.T==1], y=self.Y[self.T==1])


        null_preds = np.empty_like(self.Y, dtype=float)
        null_preds[self.T==0] = self.null_mean_model.loo_predictions()
        null_preds[self.T==1] = self.null_mean_model.predict(X_pred=self.X[self.T==1])


        treat_preds = np.empty_like(self.Y, dtype=float)
        treat_preds[self.T==1] = self.treat_mean_model.loo_predictions()
        treat_preds[self.T==0] = self.treat_mean_model.predict(X_pred=self.X[self.T==0])


        pi_star = np.clip(self.pi_star, a_min=0.01, a_max=0.99)
        alt_preds = (1./pi_star)*(treat_preds - (1 - pi_star)*null_preds)
        ite_hat = alt_preds - null_preds
        return(ite_hat)

    def calculate_fdr(self, T:np.ndarray, H:np.ndarray, fdr_levels:np.ndarray, empirical_control:bool=False):
        null_posterior = self.null_posterior.copy()

        ## Get scores on treated
        null_posterior_treated = null_posterior[T==1]
        H_treated = H[T==1]
        idx = np.argsort(null_posterior_treated)
        n_treated = idx.shape[0]
        null_posterior_treated = null_posterior_treated[idx]
        H_treated = H_treated[idx]
        treated_running_average = np.cumsum(null_posterior_treated)/np.arange(1,n_treated+1)

        n_pos = np.sum(H_treated)

        fdr_observed = np.zeros_like(fdr_levels)
        power_observed = np.zeros_like(fdr_levels)
        if empirical_control:
            ## If using empirical control, get scores on untreated as well
            null_posterior_null = null_posterior[T==0]
            H_null = H[T==0]
            idx = np.argsort(null_posterior_null)
            n_null = idx.shape[0]
            null_posterior_null = null_posterior_null[idx]
            H_null = H_null[idx]
            null_running_average = np.cumsum(null_posterior_null)/np.arange(1,n_null+1)

            ## For each fdr level, how many treated v.s. untreated selections were made?
            n_treat_selected = np.array([np.sum(treated_running_average<=alpha) for alpha in fdr_levels])
            n_null_selected = np.array([np.sum(null_running_average<=alpha) for alpha in fdr_levels])

            ## Conservative fdr estimate is # null selected/ # treated selected
            fdr_estimate = n_null_selected/np.clip(n_treat_selected, 1, np.inf)
            
            for i, alpha in enumerate(fdr_levels):
                ## Locate largest fdr level where estimate does not exceed desired alpha.
                fdr_estimate_idx = np.where(fdr_estimate <= alpha)[0]
                valid_idx = fdr_estimate_idx[fdr_estimate_idx<=i]
                if len(valid_idx)>0:
                    j = np.max(valid_idx)
                    alpha_j = fdr_levels[j]
                else:
                    alpha_j = -0.1 ## No valid selections can be made

                ## Select points below this value
                mask = treated_running_average<=alpha_j
                num_sel = np.sum(mask)
                num_neg = num_sel - np.sum(H_treated[mask])
                fdr_observed[i] = 0 if num_sel==0 else num_neg/num_sel
                power_observed[i] = np.sum(H_treated[mask])/n_pos
        else:
            for i, alpha in enumerate(fdr_levels):
                mask = treated_running_average<=alpha
                num_sel = np.sum(mask)
                num_neg = num_sel - np.sum(H_treated[mask])
                fdr_observed[i] = 0 if num_sel==0 else num_neg/num_sel
                power_observed[i] = np.sum(H_treated[mask])/n_pos

        return(fdr_observed, power_observed)
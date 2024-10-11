'''
Causal Two-Groups under the additive error assumptions:

Y = f_H(X) + epsilon
'''
import numpy as np
from causal2groups.predictive_recursion import estimate_density_dynamic, GridDistribution1D
from causal2groups.kernel_ridge import KernelRidgeRegression

def fit_prior_alternative_via_moment_matching(alternate_model:KernelRidgeRegression,
                                               X_treated:np.ndarray,
                                               y_treated_treatpreds:np.ndarray, 
                                               y_treated_nullpreds:np.ndarray, 
                                               max_steps:int,
                                               tol:float,
                                               min_prob:float,
                                               rng:np.random.Generator):
    
    N = y_treated_treatpreds.shape[0]

    ## Randomly initialize prior probabilities
    prior_probs = rng.uniform(low=0.4, high=0.6, size=N)

    err = np.inf
    step = 0
    while (step < max_steps) and (err > tol):
        prev_prior_probs = prior_probs.copy()
        
        ## mu_t = (1-pi) mu_0 + pi mu_1
        ## --> mu_1 = (mu_t - ((1-pi) mu_0))/pi
        y_treated_alt_targets = (y_treated_treatpreds - (1-prior_probs)*y_treated_nullpreds)/prior_probs
        alternate_model.fit_via_gcv(X=X_treated, 
                                    y=y_treated_alt_targets, 
                                    keep_eigen_lookup=True, 
                                    recalculate_eigen_lookup=False)
                                    
        y_treated_altpreds = alternate_model.loo_predictions()


        ## --> pi = (mu_t -  mu_0)/(mu_1 - mu_0)
        prior_probs = (y_treated_treatpreds - y_treated_nullpreds)/(y_treated_altpreds - y_treated_nullpreds)
        prior_probs = np.clip(prior_probs, a_min=min_prob, a_max=(1.-min_prob))

        ## Test convergence
        err = np.linalg.norm(prev_prior_probs - prior_probs)
        # print("step {}, err {}".format(step, err))
        step += 1
        
class KernelAdditiveCausal2G:
    def __init__(self, 
                 kernel_bandwidth_neighbors:list,
                 kernel_reg_params:list,
                 seed:int,
                 nbins=200, 
                 max_nbins=10000,
                 max_opt_steps=200, 
                 min_prob:float=0.01,
                 tol=1e-6,
                 verbose=False):
        '''

        nbins: The number of predictive recursion bins to use to approximate
               the null and alternative distributions.
               Default: 1000

        max_opt_steps: Maximum number of steps for alternating optimization.

        tol: Numerical tolerance threshold for convergence checking.
             Default: 1e-6

        verbose: If true, prints details during fitting.
        '''

        self.null_model = KernelRidgeRegression(bandwidth_neighbors=kernel_bandwidth_neighbors, 
                                                reg_params=kernel_reg_params)
        self.alternative_model = KernelRidgeRegression(bandwidth_neighbors=kernel_bandwidth_neighbors, 
                                                       reg_params=kernel_reg_params)
        self.treatment_model = KernelRidgeRegression(bandwidth_neighbors=kernel_bandwidth_neighbors, 
                                                     reg_params=kernel_reg_params)

        self.nbins = nbins
        self.max_nbins = max_nbins
        self.max_opt_steps = max_opt_steps
        self.min_prob = min_prob
        self.tol = tol
        self.verbose = verbose

        self.rng = np.random.default_rng(seed)

    def fit(self, X:np.ndarray, Y:np.ndarray, T:np.ndarray):
        if self.verbose:
            print('Fitting null model')

        self.null_model.fit_via_gcv(X=X[T==0], 
                                    y=Y[T==0])

        self.y_nullpreds = np.empty_like(Y, dtype=float)
        self.y_nullpreds[T==0] = self.null_model.loo_predictions()
        self.y_nullpreds[T==1] = self.null_model.predict(X_pred=X[T==1])
        
        if self.verbose:
            print('Fitting residual distribution')
        
        self.residual_dist = estimate_density_dynamic(y=(Y[T==0]-self.y_nullpreds[T==0]), 
                                                      nbins=self.nbins, 
                                                      max_nbins=self.max_nbins)

        if self.verbose:
            print('Fitting treatment model')

        self.treatment_model.fit_via_gcv(X=X[T==1], 
                                         y=Y[T==1])
        

        self.y_treatpreds = np.empty_like(Y, dtype=float)
        self.y_treatpreds[T==1] = self.treatment_model.loo_predictions()
        self.y_treatpreds[T==0] = self.treatment_model.predict(X_pred=X[T==0])

        if self.verbose:
            print('Fitting alternative and prior')

        fit_prior_alternative_via_moment_matching(self.alternative_model,
                                                  X_treated=X[T==1], 
                                                  y_treated_treatpreds=self.y_treatpreds[T==1], 
                                                  y_treated_nullpreds=self.y_nullpreds[T==1], 
                                                  max_steps=self.max_opt_steps, 
                                                  min_prob=self.min_prob,
                                                  tol=self.tol, 
                                                  rng=self.rng)


        self.y_altpreds = np.empty_like(Y, dtype=float)
        self.y_altpreds[T==1] = self.alternative_model.loo_predictions()
        self.y_altpreds[T==0] = self.alternative_model.predict(X_pred=X[T==0])


        prior_probs = (self.y_treatpreds - self.y_nullpreds)/(self.y_altpreds - self.y_nullpreds)
        self.prior_probs = np.clip(prior_probs, a_min=self.min_prob, a_max=(1.-self.min_prob))

        if self.verbose:
            print('Compute posterior probabilities')

        null_likelihood = np.clip(self.residual_dist.pdf(Y - self.y_nullpreds), a_min=1e-4, a_max=None)
        alt_likelihood = np.clip(self.residual_dist.pdf(Y - self.y_altpreds), a_min=1e-4, a_max=None)
        denominator = ((1-self.prior_probs)*null_likelihood) + self.prior_probs*alt_likelihood
        null_posterior = (1-self.prior_probs)*null_likelihood/denominator

        self.null_posterior = np.clip(null_posterior, a_min=0.0, a_max=1.0)

    def calculate_fdr(self, T:np.ndarray, H:np.ndarray, fdr_levels:np.ndarray, empirical_control:bool=False):
        T = T.astype(np.bool_)
        null_posterior = self.null_posterior.copy()

        ## Get scores on treated
        null_posterior_treated = null_posterior[T]
        H_treated = H[T]
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
            null_posterior_null = null_posterior[~T]
            H_null = H[~T]
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
                j = np.max(fdr_estimate_idx[fdr_estimate_idx<=i])
                alpha_j = fdr_levels[j]

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
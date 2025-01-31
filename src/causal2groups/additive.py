import numpy as np
from jax import jit
import jax.numpy as jnp
from scipy.special import expit
from causal2groups.predictive_recursion import estimate_density_dynamic
from causal2groups.kernel_ridge import KernelRidgeRegression
from causal2groups.rff_regression import RFFRegression
from causal2groups.utils import posterior_selection_fdr, posterior_selection, posterior_selection_empirical_control
from interpax import Interpolator1D
from tqdm import trange

def fit_alternative_prior_via_em(X_treat:np.ndarray, 
                                 Y_treat:np.ndarray, 
                                 Y_treat_nullpreds:np.ndarray, 
                                 log_residual_dist:Interpolator1D, 
                                 alternative_model:RFFRegression, 
                                 prior_model:RFFRegression, 
                                 max_em_steps:int,
                                 tol:float,
                                 rng:np.random.Generator, 
                                 verbose:bool):
    
    nullmodel_probs = np.exp(log_residual_dist(Y_treat - Y_treat_nullpreds))
    
    ## Set initial weights
    low_q = np.quantile(nullmodel_probs, 0.25)
    high_q = np.quantile(nullmodel_probs, 0.75)
    alt_post_weights = rng.uniform(low=0.3, high=.7, size=Y_treat.shape)
    alt_post_weights[nullmodel_probs<low_q] = 0.9
    alt_post_weights[nullmodel_probs>high_q] = 0.1

    @jit
    def alternative_loss_fn(prediction, y, weights): ## Weighted likelihood under residual distribution
        log_likelihood = log_residual_dist(y-prediction)
        neg_weighted_ll = -jnp.sum(weights*log_likelihood)
        return(neg_weighted_ll)

    @jit
    def prior_loss_fn(prediction, y, weights): ## Cross-entropy loss with logits, weights are ignored
        loss = jnp.sum((1-y)*prediction + jnp.logaddexp(0,-prediction))
        return(loss)
    

    for _ in trange(max_em_steps, disable=(not verbose)):
        prev_weights = alt_post_weights.copy()
        
        ## Fit alternative model 
        alternative_model.fit_via_cv(X=X_treat, y=Y_treat, loss=alternative_loss_fn, weights=alt_post_weights, verbose=False)
        y_treat_altpreds = alternative_model.cv_preds
        altmodel_probs = np.exp(log_residual_dist(Y_treat - y_treat_altpreds))

        ## Fit prior model
        prior_model.fit_via_cv(X=X_treat, y=alt_post_weights, loss=prior_loss_fn, verbose=False)
        prior_treat = expit(prior_model.cv_preds)
        
        ## Recalculate weights
        alt_post_weights = prior_treat*altmodel_probs/(prior_treat*altmodel_probs + (1-prior_treat)*nullmodel_probs)

        ## How much have the weights shifted?
        err = np.mean(np.abs(alt_post_weights - prev_weights))
        if err < tol:
            break


class AdditiveCausal2G:
    def __init__(self,
                 n_covariates:int,
                 rff_dims:int,
                 kernel_n_bandwidths:int,
                 kernel_reg_params:list,
                 seed:int,
                 nbins=200, 
                 max_nbins=10000,
                 max_em_steps=20, 
                 tol=1e-2,
                 verbose=False):
        '''

        nbins: The number of predictive recursion bins to use to approximate
               the null and alternative distributions.
               Default: 1000

        tol: Numerical tolerance threshold for convergence checking.
             Default: 1e-6

        verbose: If true, prints details during fitting.
        '''

        self.null_model = KernelRidgeRegression(n_bandwidths=kernel_n_bandwidths, 
                                                reg_params=kernel_reg_params)
        
        self.alternative_model = RFFRegression(n_bandwidths=kernel_n_bandwidths, 
                                       input_dim=n_covariates, 
                                       rff_dims=rff_dims, 
                                       include_identity=True, 
                                       seed=seed)
        
        self.prior_model = RFFRegression(n_bandwidths=kernel_n_bandwidths, 
                                       input_dim=n_covariates, 
                                       rff_dims=rff_dims, 
                                       include_identity=True, 
                                       seed=seed)
        
        self.nbins = nbins
        self.max_nbins = max_nbins
        self.max_em_steps = max_em_steps
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
        
        residual_dist = estimate_density_dynamic(y=(Y[T==0]-self.y_nullpreds[T==0]), 
                                                      nbins=self.nbins, 
                                                      max_nbins=self.max_nbins)

        x = residual_dist.bins
        y = residual_dist.w
        ymin = np.min(y)
        span = x[-1]-x[0]

        self.log_residual_dist = Interpolator1D(x=np.concatenate([[x[0]-span], x, [x[-1]+span]]),
                                                f=np.log(np.concatenate([[ymin*(1e-2)],y, [ymin*(1e-2)]])), 
                                                method="linear", extrap=True)

        if self.verbose:
            print('Fitting alternative and prior')

        fit_alternative_prior_via_em(X_treat=X[T==1], 
                                     Y_treat=Y[T==1], 
                                     Y_treat_nullpreds=self.y_nullpreds[T==1], 
                                     log_residual_dist=self.log_residual_dist,
                                     alternative_model=self.alternative_model, 
                                     prior_model=self.prior_model, 
                                     max_em_steps=self.max_em_steps,
                                     tol=self.tol,
                                     rng=self.rng, 
                                     verbose=self.verbose)

        self.y_altpreds = np.empty_like(Y, dtype=float)
        self.y_altpreds[T==1] = self.alternative_model.cv_preds
        self.y_altpreds[T==0] = self.alternative_model.predict(X_pred=X[T==0])


        self.prior_probs = np.empty_like(Y, dtype=float)
        self.prior_probs[T==1] = self.prior_model.cv_preds
        self.prior_probs[T==0] = self.prior_model.predict(X_pred=X[T==0])
        self.prior_probs = expit(self.prior_probs)

        null_likelihood = np.exp(self.log_residual_dist(Y - self.y_nullpreds))
        alt_likelihood = np.exp(self.log_residual_dist(Y - self.y_altpreds))
        self.null_posterior = (1-self.prior_probs)*null_likelihood/(self.prior_probs*alt_likelihood + (1-self.prior_probs)*null_likelihood)
        self.null_posterior = np.clip(self.null_posterior, a_min=0.0, a_max=1.0)

    def predict_ite(self):
        ite_hat = self.y_altpreds - self.y_nullpreds
        return(ite_hat)
    
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
    
'''
Causal Two-Groups under the additive error assumptions:

Y = f_H(X) + epsilon
'''
import numpy as np
from causal2groups.utils import create_folds, train_test_split, select_hyperparams
from causal2groups.predictive_recursion import estimate_density, generate_bins, GridDistribution1D

class AdditiveCausal2G:
    def __init__(self, null_model=None, prior_model=None,
                        null_hyperparams=None, prior_hyperparams=None,
                        nfolds=5, nbins=200, max_nbins=10000,
                        max_em_steps=10, max_inner_steps=10, tol=1e-6,
                        verbose=False):
        '''
        null_model: The predictive model for the mean of the null outcomes.
                    This must follow the sklearn conventions of having a
                    fit(X, y) function and a predict(X) function.
                    Default: Bayesian linear regression with ARD prior

        prior_model: The predictive model for the mixture prior on coming from
                     the alternative (H=1) vs. the null. (H=0). Must follow the
                     sklearn convention of having a fit(X, y) function and a
                     predict_proba(X) function.
                     Default: L2-regularized logistic regression with CV grid
                              search for the regularization penalty weight.

        null_hyperparams: The hyperparameter grid to search over when fitting
                          the null mean outcome model. The model must accept
                          these parameters via **kwargs in fit(X, y)

        prior_hyperparams: The hyperparameter grid to search over when fitting
                           the mixture prior model. The model must accept these
                           parameters via **kwargs in fit(X, y)

        nfolds: The number of cross-validation folds to use when selecting
                hyperparameters for the null and prior models.
                Default: 5

        nbins: The number of predictive recursion bins to use to approximate
               the null and alternative distributions.
               Default: 1000

        max_em_steps: Maximum number of steps for the expectation-maximization
                      algorithm used to simultaneously estimate the alternative
                      density and prior model. This is for the outer loop that
                      runs predictive recursion EM (PR-EM). The E-step is the
                      PR routine for the alternative. The M-step is to estimate
                      the predictive prior model.
                      Default: 20

        max_inner_steps: Maximum number of steps for the inner loop of the
                         PR-EM algorithm. The E-step is calculating the
                         expected posterior probability of each Z-statistic
                         coming from the alternative. The M-step is fitting the
                         weighted logistic regression problem.
                         Default: 100

        tol: Numerical tolerance threshold for convergence checking.
             Default: 1e-6

        verbose: If true, prints details during fitting.
        '''
        # Default to a bayesian linear regression model for null predictor
        if null_model is None:
            from sklearn.linear_model import BayesianRidge
            null_model = BayesianRidge()
            # from causal2groups.conditional_expectation import NeuralPredictionModel
            # null_model = NeuralPredictionModel(file_checkpoints=False)
        self.null_model = null_model
        self.null_hyperparams = null_hyperparams

        if prior_model is None:
            from causal2groups.utils import LogisticRegressionPredictor
            prior_model = LogisticRegressionPredictor()
            prior_hyperparams = {'lam': list(np.exp(np.linspace(np.log(1e-4), np.log(1e4), 20))[::-1])}
            # from causal2groups.conditional_discrete import BinaryClassifierModel
            # prior_model = BinaryClassifierModel(file_checkpoints=False)
        self.prior_model = prior_model
        self.prior_hyperparams = prior_hyperparams

        self.nfolds = nfolds
        self.nbins = nbins
        self.max_nbins = max_nbins
        self.max_em_steps = max_em_steps
        self.max_inner_steps = max_inner_steps
        self.tol = tol
        self.verbose = verbose

    def fit(self, X:np.ndarray, Y:np.ndarray, T:np.ndarray, fdr=0.1):
        '''Fits an additive causal two-groups model and performs selection on
        the treated population with control of the FDR at the target level.'''
        if self.verbose:
            print('Fitting null model')

        T = T.astype(np.bool_)
        # Standardize X and Y
        self.Y_mean = Y[~T].mean()
        self.Y_std = Y[~T].std()
        Y = (Y - self.Y_mean) / self.Y_std

        # Fit the null model and estimate the null density.
        self.estimate_null(X, Y, T)

        if self.verbose:
            print('Fitting alternative and prior via nested EM.')

        # Estimate the alternative and fit the predictive prior model.
        #self.estimate_alternative_and_prior(X, Y, T)
        self.estimate_alternative_and_prior_temp(X, Y, T)

        if self.verbose:
            print('Selecting treated individuals at the alpha={} FDR level.'.format(fdr))

        # Select the results at the alpha level
        self.select(T, fdr)

        # Predict counterfactuals:
        # 1) P(Z | X, do(t=1)) [What will the outcome be if I treat this patient?]
        # 2) P(\\hat{h}=1 and h=1 | X=x, do(t=1); fdr) [Will I be able to detect treatment effect at the target FDR level?]
        self.predict_counterfactuals(fdr)

        self.null_posterior = np.clip(1. - self.posterior_prob, 0, 1)

    def estimate_null(self, X, Y, T):
        '''Estimates a null distribution for the T=0 (no treatment).
        The model must specify a fit(X, y) function and a predict(X) function.'''
        X_treatment, Y_treatment = X[T], Y[T]
        X_control, Y_control = X[~T], Y[~T]
        
        # Split the data into nfolds
        folds = create_folds(X_control, self.nfolds)

        # Fit the model on each n-1 fold collection and predict on the held out fold
        null_errors = np.zeros(X_control.shape[0])
        for fold in folds:
            # Split into train and test for this fold
            (X_train, Y_train), (X_test, Y_test) = train_test_split(fold, X_control, Y_control)

            # Fit the null model to the training indices
            if self.null_hyperparams is None:
                self.null_model.fit(X_train, Y_train)
            else:
                # Use CV to select the hyperparameters with MSE loss
                best_hyperparams = select_hyperparams(self.null_model, X_train, Y_train, 'mse', self.null_hyperparams, nfolds=self.nfolds)
                self.null_model.fit(X_train, Y_train, **best_hyperparams)

            # Predict on held out data
            null_errors[fold] = Y_test - self.null_model.predict(X_test)

        # Save the distribution of predictive errors (estimates of noise)
        self.Z = np.zeros(T.shape)
        self.Z[~T] = null_errors

        # Fit the predictive model on the entire control population
        if self.null_hyperparams is None:
            self.null_model.fit(X_control, Y_control)
        else:
            # Use CV to select the hyperparameters
            self.best_null_hyperparams = select_hyperparams(self.null_model, X_control, Y_control, 'mse', self.null_hyperparams, nfolds=self.nfolds)
            self.null_model.fit(X_control, Y_control, **self.best_null_hyperparams)

        # Predict E[Y | do(T=0)] on all data
        self.Y_control_pred = self.null_model.predict(X)

        # Get the prediction errors for the treated
        self.Z[T] = Y_treatment - self.Y_control_pred[T]

        import warnings
        finished = False
        self.null_dist = None
        while (not finished) and (self.nbins < self.max_nbins):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    # Get the bins for estimating the null distribution via predictive recursion
                    bins = generate_bins(self.Z, self.nbins)

                    # Estimate the null distribution using predictive recursion
                    self.null_dist = estimate_density(self.Z[~T], bins=bins)['dist']

                    finished = True
            except:
                warnings.warn('Insufficient bins for residuals. Doubling from {} to {}'.format(self.nbins, 2*self.nbins))
                self.nbins = 2 * self.nbins
                
        if self.null_dist is None:
            bins = generate_bins(self.Z, self.max_nbins)
            self.null_dist = estimate_density(self.Z[~T], bins=bins)['dist']
        
    # TEMP
    def estimate_alternative_and_prior_temp(self, X, Y, T):
        '''Takes in a set of estimated Z scores from a treated population, as well
        as a null distribution estimate f0, and estimates BOTH the marginal
        alternative effect distribution and the prior probability of treatment
        effect.'''
        # Cache the likelihoods
        f0 = self.null_dist.pdf(self.Z)
        #self.posterior_prob = np.ones(self.Z.shape)
        self.posterior_prob = np.abs(self.null_dist.cdf(self.Z) - 0.5)*2 # 1 minus two-sided p-value of the null

        prev_delta = 1
        prev_outer = np.array(self.posterior_prob[T])
        sweeporder = None
        for outer_step in range(self.max_em_steps):
            if self.verbose:
                print('\tStep #{}'.format(outer_step+1))

            # Outer E-step: predictive recursion to estimate the alternative
            # TODO: should this be estimate_density? -- maybe i did this on accident?
            # alt_fit = estimate_mixture(self.Z[T], self.null_dist, bins=self.null_dist.bins, weights=self.posterior_prob[T], sweeporder=sweeporder)
            alt_fit = estimate_density(self.Z[T], bins=self.null_dist.bins, weights=self.posterior_prob[T], sweeporder=sweeporder)
            alt_dist, sweeporder = alt_fit['dist'], alt_fit['sweeporder']
            f1 = alt_dist.pdf(self.Z)

            # Create the helper object to make fitting the prior via EM easier.
            em_model = EMPrior(self.prior_model, self.max_inner_steps, self.tol)

            # The ``labels'' are the likelihood values for the null and alternative
            labels = np.array([f0,f1]).T

            # Use CV to select the hyperparameters, if any.
            if self.prior_hyperparams is not None:
                def cv_loss(model, X_cv, Y_cv):
                    '''Cross-entropy loss'''
                    f0_cv, f1_cv = Y_cv.T
                    h = model.predict_proba(X_cv)
                    return -np.log(h*f1_cv + (1-h)*f0_cv).mean()
                best_hyperparams = select_hyperparams(em_model, X[T], labels[T], cv_loss, self.prior_hyperparams, nfolds=self.nfolds)
                em_model.fit(X[T], labels[T], **best_hyperparams)
            else:
                em_model.fit(X[T], labels[T])

            # Get the prior probabilities for both the control and treated populations
            prior_prob = self.prior_model.predict_proba(X)
            posterior_prob = prior_prob*f1 / ((1-prior_prob)*f0 + prior_prob*f1)

            # Check for convergence
            delta = np.linalg.norm(posterior_prob[T] - prev_outer)

            if self.verbose:
                print('\tDelta: {:.6f}'.format(delta))
                print()

            # Stop if we've converged
            if delta <= self.tol:
                break
            prev_outer = np.array(posterior_prob[T])

            # FLAG: early stopping to handle weird spike
            if (outer_step >= 0.8*self.max_em_steps) and delta > 2*prev_delta:
                break
            else:
                self.alt_dist = alt_dist
                self.prior_prob = prior_prob
                self.posterior_prob = posterior_prob
                prev_delta = delta

    def estimate_alternative_and_prior(self, X, Y, T):
        '''Takes in a set of estimated Z scores from a treated population, as well
        as a null distribution estimate f0, and estimates BOTH the marginal
        alternative effect distribution and the prior probability of treatment
        effect.'''
        # Cache the likelihoods
        f0 = self.null_dist.pdf(self.Z)
        #self.posterior_prob = np.ones(self.Z.shape)
        self.posterior_prob = np.abs(self.null_dist.cdf(self.Z) - 0.5)*2 # 1 minus two-sided p-value of the null

        prev_outer = np.array(self.posterior_prob[T])
        sweeporder = None
        for outer_step in range(self.max_em_steps):
            if self.verbose:
                print('\tStep #{}'.format(outer_step+1))

            # Outer E-step: predictive recursion to estimate the alternative
            # TODO: should this be estimate_density? -- maybe i did this on accident?
            # alt_fit = estimate_mixture(self.Z[T], self.null_dist, bins=self.null_dist.bins, weights=self.posterior_prob[T], sweeporder=sweeporder)
            alt_fit = estimate_density(self.Z[T], bins=self.null_dist.bins, weights=self.posterior_prob[T], sweeporder=sweeporder)
            self.alt_dist, sweeporder = alt_fit['dist'], alt_fit['sweeporder']
            f1 = self.alt_dist.pdf(self.Z)

            # Create the helper object to make fitting the prior via EM easier.
            em_model = EMPrior(self.prior_model, self.max_inner_steps, self.tol)

            # The ``labels'' are the likelihood values for the null and alternative
            labels = np.array([f0,f1]).T

            # Use CV to select the hyperparameters, if any.
            if self.prior_hyperparams is not None:
                def cv_loss(model, X_cv, Y_cv):
                    '''Cross-entropy loss'''
                    f0_cv, f1_cv = Y_cv.T
                    h = model.predict_proba(X_cv)
                    return -np.log(h*f1_cv + (1-h)*f0_cv).mean()
                best_hyperparams = select_hyperparams(em_model, X[T], labels[T], cv_loss, self.prior_hyperparams, nfolds=self.nfolds)
                em_model.fit(X[T], labels[T], **best_hyperparams)
            else:
                em_model.fit(X[T], labels[T])

            # Get the prior probabilities for both the control and treated populations
            self.prior_prob = self.prior_model.predict_proba(X)
            self.posterior_prob = self.prior_prob*f1 / ((1-self.prior_prob)*f0 + self.prior_prob*f1)

            # Check for convergence
            delta = np.linalg.norm(self.posterior_prob[T] - prev_outer)

            if self.verbose:
                print('\tDelta: {:.6f}'.format(delta))
                print()

            # Stop if we've converged
            if delta <= self.tol:
                break
            prev_outer = np.array(self.posterior_prob[T])

    def select(self, T, fdr, criteria=True):
        '''Select results on treated population with FDR control at the target level.'''
        indices = np.arange(T.shape[0])[T]
        W_treated = self.posterior_prob[T]
        order = np.argsort(W_treated)[::-1] # Step down from the top
        nrejects = np.arange(indices.shape[0]+1) # Consider every rejection point, including no rejections
        cumfdr = (nrejects - np.concatenate([[0],W_treated[order].cumsum()])) / nrejects.clip(1,np.inf) # Get all the FDR levels
        # Add selection trick
        if criteria:
            self.selected = [i for i in indices[order[:nrejects[cumfdr <= fdr].max()]] if self.posterior_prob[i] >= (1-fdr)**2] # add alpha-budget for posterior
        else:
            self.selected = indices[order[:nrejects[cumfdr <= fdr].max()]]
        return self.selected


    def predict_counterfactuals(self, fdr):
        '''Given the model, calculate predictive estimates:
        1) P(Z | X, do(t=1)) [What will the outcome be if I treat this patient?]
        2) P(\\hat{h}=1 and h=1 | X=x, do(t=1); fdr) [Will I be able to detect treatment effect at the target FDR level?]
        '''        
        # Marginalize out prior probability to get P(Y | X=x, do(t=1))
        f0_likelihood = self.null_dist.pdf(self.alt_dist.bins)
        f1_likelihood = self.alt_dist.pdf(self.alt_dist.bins)
        outcome_probs = f0_likelihood[None] * (1-self.prior_prob)[:,None] + f1_likelihood[None] * self.prior_prob[:,None]
        self.Y_treatment_pred = [GridDistribution1D((self.alt_dist.bins+y0)*self.Y_std+self.Y_mean, o) for o,y0 in zip(outcome_probs, self.Y_control_pred)]

        # Calculate the distribution over local FDR thresholds
        # to see if we can detect an effect from treatment
        # P(\hat{h}=1 and h=1 | X=x, do(t=1); fdr)
        posterior_levels = f1_likelihood[None]*self.prior_prob[:,None] / outcome_probs # P(h=1 | X, Z, do(t=1))
        self.fdr_levels = posterior_levels >= (1-fdr)
        self.detections = (f1_likelihood[None]*self.fdr_levels).sum(axis=1) / f1_likelihood.sum()
        # alpha_probs = np.sum((z_grid[1:] - z_grid[0:-1])[None,:] * (alpha_levels[:,1:] + alpha_levels[:,0:-1]) / 2., axis=1) / (z_grid[-1] - z_grid[0]) # Trapezoid method


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

class EMPrior:
    def __init__(self, model, max_steps, tol):
        self.model = model
        self.max_steps = max_steps
        self.tol = tol

    def fit(self, X, Y, **kwargs):
        '''EM inner loop: Fit the prior probability model given a fixed
        alternative distribution f1.'''
        f0, f1 = Y.T
        prev = np.inf
        W = np.zeros(X.shape[0])
        W = f1 / (f0 + f1) # initialize with uniform prior
        for step in range(self.max_steps):
            # M-step: run logistic regression
            if kwargs is None:
                self.model.fit(X, W)
            else:
                self.model.fit(X, W, **kwargs)

            # E-step: update the local FDR (posterior) weights
            G = self.model.predict_proba(X)
            W = G*f1 / ((1-G)*f0 + G*f1)

            # Check for convergence
            delta = np.linalg.norm(W - prev)
            if delta <= self.tol:
                return
            prev = np.array(W)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

def plot_observed_outcomes(Y, T, prefix='additive'):
    import matplotlib.pyplot as plt
    plt.hist(Y[~T], color='gray', bins=np.linspace(Y.min(), Y.max(), 50), alpha=0.5, label='Untreated', weights=np.ones(Y[~T].shape[0])/Y[~T].shape[0])
    plt.hist(Y[T], color='purple', bins=np.linspace(Y.min(), Y.max(), 50), alpha=0.7, label='Treated',
                                        weights=np.ones(Y[T].shape[0])/Y[T].shape[0])
    plt.xlabel('Observed outcomes', fontsize=18)
    plt.legend(loc='upper right')
    plt.savefig('plots/{}-outcome-hists.pdf'.format(prefix), bbox_inches='tight')
    plt.close()

def plot_dist_estimates(model, T, H, prefix='additive', true_null=None):
    import matplotlib.pyplot as plt
    Z_null = model.Z[~T]
    plt.hist(Z_null, color='gray', bins=50, alpha=0.5, label='Untreated', weights=np.ones(Z_null.shape[0])/Z_null.shape[0])
    
    treated_no_effect = T & (~H)
    Z_no_effect = model.Z[treated_no_effect]
    plt.hist(Z_no_effect, color='purple', bins=50, alpha=0.7, label='Treated with no effect',
                               weights=np.ones(Z_no_effect.shape[0])/Z_no_effect.shape[0])

    treated_some_effect = T & H
    Z_some_effect = model.Z[treated_some_effect]
    plt.hist(Z_some_effect, color='orange', bins=50, alpha=0.7, label='Treated with effect',
                                        weights=np.ones(Z_some_effect.shape[0])/Z_some_effect.shape[0])

    # Plot the true null and the estimated null
    prob_z_grid = model.null_dist.pdf(model.null_dist.bins)
    if true_null is not None:
        prob_true_z = true_null.pdf(model.null_dist.bins)
        plt.plot(model.null_dist.bins, prob_true_z / prob_true_z.sum(), color='black', label='True $f_0$')
    plt.plot(model.null_dist.bins, prob_z_grid / prob_z_grid.sum(), color='darkgray', label='Estimated $f_0$')

    # Plot the estimated alternative
    prob_z_alt = model.alt_dist.pdf(model.null_dist.bins)
    plt.plot(model.null_dist.bins, prob_z_alt / prob_z_alt.sum(), color='orange', label='Estimated $f_1$')

    # Save the figure
    plt.xlabel('Residual $\\hat{r}$', fontsize=18)
    plt.legend(loc='upper left')
    plt.savefig('plots/{}-alt-hists.pdf'.format(prefix), bbox_inches='tight')
    plt.close()

def plot_prior_probs(model, truth, T, prefix='additive'):
    import matplotlib.pyplot as plt
    # Plot the true vs estimated prior probabilities of seeing an effect given treatment
    plt.scatter(truth[~T], model.prior_prob[~T], color='gray', alpha=0.5, label='Untreated')
    plt.scatter(truth[T], model.prior_prob[T], color='orange', alpha=0.5, label='Treated')
    plt.plot([0,1],[0,1],ls='--',color='red')
    plt.xlabel('True effect prior', fontsize=18)
    plt.ylabel('Estimated effect prior', fontsize=18)
    plt.savefig('plots/{}-effect-propensities.pdf'.format(prefix), bbox_inches='tight')
    plt.close()

def plot_posteriors(model, T, H, prefix='additive'):
    import matplotlib.pyplot as plt
    control = ~T
    W_control = model.posterior_prob[control]
    plt.hist(W_control, color='gray', bins=20, alpha=0.5, label='Untreated', weights=np.ones(W_control.shape[0])/W_control.shape[0])

    treated_no_effect = T & (~H)
    W_no_effect = model.posterior_prob[treated_no_effect]
    plt.hist(W_no_effect, color='purple', bins=20, alpha=0.7, label='Treated with no effect',
                                        weights=np.ones(W_no_effect.shape[0])/W_no_effect.shape[0])

    treated_some_effect = T & H
    W_some_effect = model.posterior_prob[treated_some_effect]
    plt.hist(W_some_effect, color='orange', bins=20, alpha=0.7, label='Treated with effect',
                                        weights=np.ones(W_some_effect.shape[0])/W_some_effect.shape[0])
    
    # Save the figure
    plt.xlabel('Posterior P(h=1 | x, z)', fontsize=18)
    plt.legend(loc='upper left')
    plt.savefig('plots/{}-posterior-hists.pdf'.format(prefix), bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(model, Y, T, H, prefix='additive'):
    import matplotlib.pyplot as plt
    # Plot the True/False Positive/Negative confusion matrix as histograms of outcomes in each category.
    S = np.zeros(Y.shape[0], dtype=bool)
    S[model.selected] = True

    plot_bins = np.linspace(Y.min(), Y.max(), 50)
    masks = [~T, (~S) & T & (~H), (~S) & T & H, S & T & (~H), S & T & H]
    labels = ['Untreated', 'Treated True Negatives', 'Treated False Negatives', 'Treated False Positives', 'Treated True Positives']
    colors = ['0.5', '0.75', '0.65', '0.1', '0.3']
    filenames = ['control', 'tn', 'fn', 'fp', 'tp']
    for mask, label, color, filename in zip(masks, labels, colors, filenames):
        plt.hist(Y[mask], color=color, bins=plot_bins, label=label)
        plt.xlabel('Observed outcome')
        plt.legend(loc='upper right')
        plt.savefig('plots/{}-confusing-{}.pdf'.format(prefix, filename), bbox_inches='tight')
        plt.close()

def plot_detections(model, fdr, prefix='additive'):
    import matplotlib.pyplot as plt
    plt.hist(model.detections, bins=np.linspace(0,1,51))
    plt.xlabel('Probability of detecting effect at $\\alpha={}$ level'.format(fdr))
    plt.savefig('plots/{}-detections.pdf'.format(prefix), bbox_inches='tight')
    plt.close()

def plot_potential_outcomes(model, nsamples=5, prefix='additive'):
    import matplotlib.pyplot as plt
    # plot_bins = model.Y_treatment_pred[0].bins
    [plt.plot(o.bins, o.pdf(o.bins), color=str((i+1)/(nsamples+1))) for i,o in enumerate(model.Y_treatment_pred[:nsamples])]
    plt.xlabel('Potential outcome Y(1)')
    plt.ylabel('Probability')
    plt.savefig('plots/{}-potential-outcomes.pdf'.format(prefix), bbox_inches='tight')
    plt.close()




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from causal2groups.utils import ilogit
    N = 2000 # Observational samples
    P = 10 # Confounders
    fdr = 0.1 # Target FDR level
    np.random.seed(6)
    
    # Simulate some confounded data
    X = np.random.normal(0, 1/np.sqrt(P), size=(N,P)) # Observed confounders
    T_propensity = ilogit(X.dot(np.random.normal(size=P))) # Prob. of treatment assignment
    H_prob = ilogit(X.dot(np.random.normal(size=P))) # Prob. of effect given do(T=1)
    X_effect = np.random.standard_t(3,size=P) # Confounder effects
    T = (np.random.random(size=N) <= T_propensity) # Treatment assignment
    T_effect_size = np.random.normal(2, size=N) # Homogeneous treatment effect, if it has one
    Y = np.log1p(np.exp(X.dot(X_effect))) + np.random.standard_t(5, size=N) # Null outcomes
    H = (np.random.random(size=N) <= H_prob) # Did this treatment have any effect?
    Y[T] += H[T] * T_effect_size[T] # Treatment effects are additive if present

    # Run the additive model
    model = AdditiveCausal2G(verbose=True)
    model.fit(X, Y, T, fdr=fdr)

    # Plot the results!
    from scipy.stats import t as student_t
    true_null = lambda z: student_t.pdf(z, 5)
    plot_observed_outcomes(Y, T)
    plot_dist_estimates(model, T, H, true_null=true_null)
    plot_prior_probs(model, H_prob, T)
    plot_posteriors(model, T, H)
    plot_confusion_matrix(model, Y, T, H)
    plot_detections(model, fdr)
    plot_potential_outcomes(model)





import numpy as np


def ilogit(x):
    return 1/(1+np.exp(-x))

def create_folds(X, k):
    if isinstance(X, int) or isinstance(X, np.integer):
        indices = np.arange(X)
    elif hasattr(X, '__len__'):
        indices = np.arange(len(X))
    else:
        indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) // k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds

def fold_complement(N, fold):
    '''Returns all the indices from 0 to N that are not in fold.'''
    mask = np.ones(N, dtype=bool)
    mask[fold] = False
    return np.arange(N)[mask]

def train_test_split_old(X, Y, test_indices, weights=None):
    '''Splits X and Y into train and test sets based on test_indices''' 
    train = fold_complement(X.shape[0], test_indices)
    X_train, Y_train = X[train], Y[train]
    X_test, Y_test = X[test_indices], Y[test_indices]
    if weights is not None:
        W_train, W_test = weights[train], weights[test_indices]
    return (X_train, Y_train), (X_test, Y_test)

def train_test_split(test_indices, *splittables):
    if len(splittables) == 0:
        return []
    N = len(splittables[0])
    train_indices = fold_complement(N, test_indices)
    return ([v[train_indices] for v in splittables],
            [v[test_indices] for v in splittables])

def batches(indices, batch_size, shuffle=True):
    order = np.copy(indices)
    if shuffle:
        np.random.shuffle(order)
    nbatches = int(np.ceil(len(order) / float(batch_size)))
    for b in range(nbatches):
        idx = order[b*batch_size:min((b+1)*batch_size, len(order))]
        yield idx

def dict_product(d):
    '''Get the Cartesian product of dictionary values, where each instance
    yields a dictionary with the same keys as the original dictionary and
    the values are for a single instance of the Cartesian product.'''
    from itertools import product
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    for v in product(*vals):
        yield {k: kv for k, kv in zip(keys, v)}

def mse_loss(model, X, y, weights=None, **kwargs):
    '''Mean squared error loss.'''
    if weights is None:
        weights = np.ones(X.shape[0])
    return (weights*(y - model.predict(X))**2).sum() / weights.sum()

def select_hyperparams(model, X, y, loss, hyperparams, splittables=None, nfolds=5):
    '''Select hyperparameters for a predictive model.

    model: The predictive model. Must take hyperparams and splittables via
           **kwargs in fit(X,y,**kwargs).

    loss:  The scoring loss to minimize via CV. Takes all splittables via **kwargs.
    
    hyperparams: A dictionary where each key maps to a list of possible
                 values for that parameter.

    splittables: A dictionary where each key maps to an array where the first
                 dimension is the same size as that of X and y.

    All hyperparams and splittables will be passed as named keyword parameters
    to the fit function.
    '''
    if hyperparams is None:
        # Why did you call this function if you have no hyperparameters?
        return None
    if np.all([len(v) == 0 for k, v in hyperparams.items()]):
        # If all the hyperparameters only have one option, there's no choice
        return {k: v[0] for k, v in hyperparams.items()}
    if splittables is None:
        splittables = {}

    if loss == 'mse':
        loss = mse_loss

    # Split everything into folds
    folds = create_folds(X, nfolds)
    keys = list(splittables.keys())
    to_split = [X, y] + [splittables[k] for k in keys]

    scores = []
    for fidx, fold in enumerate(folds):
        # Split into train and test for this fold
        train_splits, test_splits = train_test_split(fold, *to_split)
        (X_train, y_train), (X_test, y_test) = train_splits[:2], test_splits[:2]
        train_params = {k: v for k, v in zip(keys, train_splits[2:])}
        test_params = {k: v for k, v in zip(keys, test_splits[2:])}
        
        for hidx, cur_hyperparams in enumerate(dict_product(hyperparams)):
            # Copy over the current hyperparameters
            for k, v in cur_hyperparams.items():
                train_params[k] = v

            # Fit the model to the training data
            model.fit(X_train, y_train, **train_params)

            # Evaluate the model on testing data
            score = loss(model, X_test, y_test, **test_params)

            # Add the score to the list
            if fidx == 0:
                scores.append(np.zeros(nfolds))

            scores[hidx][fidx] = score

    scores = np.array(scores)

    # Average over all folds
    scores = scores.mean(axis=1)

    # Find the best parameters
    best = np.argmin(scores)

    # Return the best hyperparameters
    for hidx, cur_hyperparams in enumerate(dict_product(hyperparams)):
        if hidx == best:
            return cur_hyperparams

def select_hyperparams_and_fit(model, X, y, loss, hyperparams, splittables=None, nfolds=5):
    if splittables is None:
        splittables = {}
    if hyperparams is None:
        model.fit(X, y, **splittables)
    else:
        # Use CV to select the hyperparameters
        best_hyperparams = select_hyperparams(model, X, y, loss, hyperparams,
                                              splittables=splittables,
                                              nfolds=nfolds)
        # Copy over the best hyperparameters
        for k, v in best_hyperparams.items():
            splittables[k] = v

        model.fit(X, y, **splittables)


def trapezoid(x, y):
    return np.sum((x[1:] - x[0:-1]) * (y[1:] + y[0:-1]) / 2.)


class GridDistribution:
    def __init__(self, x, y):
        self.x = x
        self.y = y / trapezoid(x, y)

    def pdf(self, data):
        # Find the closest bins
        rhs = np.searchsorted(self.x, data)
        lhs = (rhs - 1).clip(0)
        rhs = rhs.clip(0, len(self.x) - 1)

        # Linear approximation (trapezoid rule)
        rhs_dist = np.abs(self.x[rhs] - data)
        lhs_dist = np.abs(self.x[lhs] - data)
        denom = rhs_dist + lhs_dist
        denom[denom == 0] = 1. # handle the zero-distance edge-case
        rhs_weight = 1.0 - rhs_dist / denom
        lhs_weight = 1.0 - rhs_weight

        return lhs_weight * self.y[lhs] + rhs_weight * self.y[rhs]


def logistic_regression_loss(X, y, w, lam, beta):
    intercept = beta[-1] if len(beta) > X.shape[1] else 0
    beta = beta[:-1] if len(beta) > X.shape[1] else beta
    preds = ilogit(X.dot(beta) + intercept).clip(1e-6,1-1e-6)
    return -(w*(y*np.log(preds) + (1-y)*np.log(1-preds))).sum() / w.sum()  + lam*(beta**2).sum()

def logistic_regression_grad(X, y, w, lam, beta):
    grad = np.zeros(len(beta))
    intercept = beta[-1] if len(beta) > X.shape[1] else 0
    beta = beta[:-1] if len(beta) > X.shape[1] else beta
    preds = ilogit(X.dot(beta) + intercept).clip(1e-6,1-1e-6)
    grad[:X.shape[1]] = (w[:,None]*X).T.dot(preds - y) / w.sum() + lam*beta
    if len(grad) > X.shape[1]:
        grad[-1] = (w*(preds - y)).sum() / w.sum()
    return grad

def logistic_barrier_loss(X, barrier, rho, beta):
    intercept = beta[-1] if len(beta) > X.shape[1] else 0
    beta = beta[:-1] if len(beta) > X.shape[1] else beta
    preds = ilogit(X.dot(beta) + intercept).clip(1e-6,1-1e-6)
    mu = preds.mean()
    if mu <= barrier:
        return 0
    return rho*(mu - barrier)**2

def logistic_barrier_grad(X, barrier, rho, beta):
    grad = np.zeros(len(beta))
    intercept = beta[-1] if len(beta) > X.shape[1] else 0
    beta = beta[:-1] if len(beta) > X.shape[1] else beta
    preds = ilogit(X.dot(beta) + intercept).clip(1e-6,1-1e-6)
    mu = preds.mean()
    if mu > barrier:
        opp_pred = 1-preds
        grad[:X.shape[1]] = X.T.dot(opp_pred-opp_pred**2) / X.shape[0]
        if len(grad) > X.shape[1]:
            grad[-1] = (opp_pred-opp_pred**2).mean()
        grad *= rho*2*(mu - barrier)
    return grad

class LogisticRegressionPredictor:
    def __init__(self, lam=0, fit_intercept=True, init_coefs=None, barrier=None, nbarrier_steps=10):
        self.lam = lam
        self.fit_intercept = fit_intercept
        self.barrier = barrier
        self.nbarrier_steps = nbarrier_steps

        if init_coefs is not None:
            self.coef_ = np.array(init_coefs)
        else:
            self.coef_ = None

    def fit(self, X, y, W=None, lam=None):
        from functools import partial
        from scipy.optimize import fmin_l_bfgs_b
        if W is None:
            W = np.ones(X.shape[0])
        if self.coef_ is None:
            self.coef_ = np.random.normal(size=X.shape[1] + int(self.fit_intercept))
        if lam is not None:
            self.lam = lam
        lr_loss = partial(logistic_regression_loss, X, y, W, self.lam)
        lr_grad = partial(logistic_regression_grad, X, y, W, self.lam)
        if self.barrier is None:
            self.coef_ = fmin_l_bfgs_b(lr_loss, self.coef_, lr_grad)[0]
        else:
            rho_grid = np.exp(np.linspace(np.log(1e-4),np.log(1e4),self.nbarrier_steps))
            for rho in rho_grid:
                lbar_loss = partial(logistic_barrier_loss, X, self.barrier, rho)
                lbar_grad = partial(logistic_barrier_grad, X, self.barrier, rho)
                step_loss = lambda beta: lr_loss(beta) + lbar_loss(beta)
                step_grad = lambda beta: lr_grad(beta) + lbar_grad(beta)
                self.coef_ = fmin_l_bfgs_b(step_loss, self.coef_, step_grad)[0]

    def predict_proba(self, X):
        intercept = self.coef_[-1] if self.fit_intercept else 0
        return ilogit(X.dot(self.coef_[:X.shape[1]]) + intercept)

def test_barrier_lr():
    N = 100
    P = 5
    X = np.random.normal(0, 1/np.sqrt(P), size=(N,P))
    barrier = 0.3
    logits = X.dot(np.random.normal(size=P))
    probs = ilogit(logits)
    y = (np.random.random(size=N) <= probs).astype(float)
    model = LogisticRegressionPredictor(barrier=barrier)
    def pi_loss(model, X_cv, Y_cv):
        pred = model.predict_proba(X_cv)
        return -(Y_cv*np.log(pred) + (1-Y_cv)*np.log(1-pred)).mean()
    hyperparams = {'lam': list(np.exp(np.linspace(np.log(1e-6), np.log(1e4), 20))[::-1])}
    # Estimate pi
    select_hyperparams_and_fit(model, X, y,
                                hyperparams=hyperparams,
                                nfolds=5,
                                loss=pi_loss)
    # model.fit(X, y)
    preds = model.predict_proba(X)
    print('Range of predictions: [{:.2f}, {:.2f}] Expectation: {:.2f} Barrier: {:.2f}'.format(preds.min(), preds.max(), preds.mean(), barrier))
    import matplotlib.pyplot as plt
    plt.scatter(probs, preds, color='black')
    plt.plot([0,1], [0,1], ls='--', color='red')
    plt.savefig('plots/test-barrier.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    test_barrier_lr()



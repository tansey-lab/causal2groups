import numpy as np

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



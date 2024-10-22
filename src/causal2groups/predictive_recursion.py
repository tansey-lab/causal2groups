### COPIED FROM ‘PRDENSITY’ PACKAGE

'''Predictive recursion-based routines for estimating marginal distributions of
z-scores as an infinite mixture of gaussians.'''
import numpy as np
from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid, cumulative_trapezoid

class GridDistribution1D:
    def __init__(self, bins, w, discrete=False):
        self.bins = bins
        
        if discrete:
            self.w = w / w.sum()
            self.w_cum = np.cumsum(self.w)
            self.grid = self.discrete_grid
            self.cdf_grid = self.discrete_cdf_grid
        else:
            self.w = w / trapezoid(w, bins)
            self.w_cum = cumulative_trapezoid(self.w, self.bins, initial=0)
            self.grid = RegularGridInterpolator((bins,), self.w, bounds_error=False, fill_value=0)
            self.cdf_grid = RegularGridInterpolator((bins,), self.w_cum, bounds_error=False, fill_value=0)

        # Quick and dirty expectation (TODO: better estimate this)
        self.expectation = (self.grid(self.bins) * self.bins).sum() / self.grid(self.bins).sum()

    def pdf(self, x):
        return self.grid(x)

    def cdf(self, x):
        if np.isscalar(x):
            return self.cdf_grid(np.array([x]))[0]
        return self.cdf_grid(x)

    def bins_expand(self, x):
        bins = self.bins
        while len(bins.shape) <= len(x.shape):
            bins = bins[None]
        x = x[...,None]
        return x, bins

    def discrete_grid(self, x):
        if np.isscalar(x):
            return np.where(self.bins == x, self.w)
        x, bins = self.bins_expand(x)
        return self.w[np.argmax(bins == x, axis=-1)]

    def discrete_cdf_grid(self, x):
        if np.isscalar(x):
            return np.where(self.bins == x, self.w_cum)
        x, bins = self.bins_expand(x)
        return self.w_cum[np.argmax(bins==x, axis=-1)]

    def mean(self):
        mean = trapezoid(y=self.grid(self.bins)*self.bins, x=self.bins)
        return(mean)
    
    def variance(self):
        sq = trapezoid(y=self.grid(self.bins)*np.square(self.bins), x=self.bins)
        var = sq-np.square(self.mean())
        return(var)

def generate_sweeps(num_sweeps, num_samples):
    '''Creates random sweeps over the data.'''
    results = []
    for sweep in range(num_sweeps):
        a = np.arange(num_samples)
        np.random.shuffle(a)
        results.append(a)
    return np.array(results)

def generate_bins(z, bins, lower_bound=None, upper_bound=None):
    # Handle multi-dimensional Z
    if len(z.shape) > 1:
        if np.isscalar(bins):
            return [generate_bins(z[...,i], bins) for i in range(z.shape[-1])]
        return [generate_bins(z[...,i], bins[i]) for i in range(z.shape[-1])]
    if np.isscalar(bins):
        # Linear grid over the range of z, with 10% overhang on each side
        z_range = z.max() - z.min()
        if lower_bound is None:
            lower_bound = -np.inf
        if upper_bound is None:
            upper_bound = np.inf
        bins = np.linspace(max(lower_bound, z.min() - z_range*0.1), min(upper_bound, z.max() + z_range*0.1), bins)
    return bins


def estimate_density(y, bins=200, weights=None, tilts=None, nsweeps=10, sweeporder=None,
                        decay=-0.67, sigma=None, sbins=30, kernel=None, prior_init=None,
                        lower_bound=None, upper_bound=None, support_bins=None, discrete=False,
                        **kwargs):
    '''Estimates a marginal density as a mixture of normals using predictive
    recursion.'''
    if sweeporder is None:
        # Create random sweeps through the dataset
        sweeporder = generate_sweeps(nsweeps, len(y))

    bins = generate_bins(y, bins, lower_bound=lower_bound, upper_bound=upper_bound)

    if sigma is None:
        if np.isscalar(sbins):
            sbins = np.linspace(1e-2, 4, sbins)
        
        # Estimate the density using a range of kernel bandwidths
        results = [estimate_density(y, sigma=s, sweeporder=sweeporder, bins=bins,
                                       kernel=kernel, nsweeps=nsweeps, weights=weights,
                                       decay=decay, sbins=sbins, prior_init=prior_init, **kwargs) for s in sbins]

        # Calculate the 'predictive recursion marginal likelihood' (PRML) of y for each bandwidth
        marginals = np.array([r['logm'] for r in results])
        marginals[np.isnan(marginals)] = -np.inf

        # Weight each density proportional to the PRML of y
        Zs = [r['z'] for r in results]
        z_logits = np.exp(marginals - np.max(marginals))
        z_probs = z_logits / z_logits.sum()
        z_hat = (z_probs[:,None] * Zs).sum(axis=0)

        # Create the new weighted distribution
        result =  results[np.argmax(marginals)]
        result['dist'] = GridDistribution1D(bins, z_hat)
        result['z'] = z_hat
        result['logm_grid'] = marginals
        result['w'] = (z_probs[:,None] * [r['w'] for r in results]).sum(axis=0)
        result['sbins'] = sbins

        return result

    if weights is None:
        # Assign equal weight to every data point
        weights = np.ones_like(y)

    if tilts is None:
        # No tilting by default
        tilts = np.ones((len(y), len(bins)))

    if kernel is None or kernel == 'normal':
        kernel = lambda x, b, s: norm.pdf(x[:,None], b[None], scale=s)

    if prior_init is None:
        # Default to uniform prior
        prior_init = lambda: np.ones(len(bins)) / (bins.max() - bins.min())

    # Initialize everything to equal weights
    w = np.zeros(bins.shape)

    # Calculate the likelihood of each y coming from a N(mu, 1) centered at this bin
    likelihoods = kernel(y, bins, sigma)

    # Sweep through the data and reweight the density using each point iteratively
    log_marginal = 0
    for sweep_idx, sweep in enumerate(sweeporder):
        w_sweep = prior_init()
        cum_weights = 0
        for i, k in enumerate(sweep):
            #step_weight = (3. + i)**decay # Each iteration contributes slightly less
            step_weight = (3. + cum_weights)**decay
            f = w_sweep * tilts[k] * likelihoods[k] # prob of z_k coming from N(bins, 1) * current prior
            m = max(1e-10, trapezoid(f, bins))
            if i < len(y):
                log_marginal += np.log(max(1e-10, m))
            w_sweep = (1. - step_weight * weights[k]) * w_sweep + step_weight * weights[k] * f/m # reweight
            cum_weights += weights[k]
        w += w_sweep / sweeporder.shape[0]

    # Get the marginal likelihood of each point to create a grid approximation
    if support_bins is None:
        support_bins = bins
    z = (w[None]*kernel(support_bins, bins, sigma)).sum(axis=1)

    if discrete:
        # Discrete distribution over the support bins
        z /= z.sum()
    else:
        # Continuous distribution over the support
        z /= trapezoid(z, support_bins)

    return {'bins': bins, 
            'support_bins': support_bins, 
            'dist': GridDistribution1D(support_bins, z, discrete=discrete),
            'w': w, 'z': z, 'logm': log_marginal, 'sweeporder': sweeporder}



def estimate_density_dynamic(y, nbins, max_nbins):
    import warnings
    finished = False
    dist:GridDistribution1D = None
    
    while (not finished) and (nbins < max_nbins):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                # Estimate the null distribution using predictive recursion
                dist = estimate_density(y, bins=nbins)['dist']
                finished = True
        except:
            warnings.warn('Insufficient bins for residuals. Doubling from {} to {}'.format(nbins, 2*nbins))
            nbins = 2 * nbins
            
    if dist is None:
        dist = estimate_density(y, bins=max_nbins)['dist']
    return(dist)

class GmmDummy:
    def __init__(self, pi, mu, sigma):
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x):
        return (self.pi[None] * norm.pdf(x[:,None], self.mu[None], self.sigma[None])).sum(axis=1)


def test_estimate_density():
    N = 300
    mu = np.array([-2, 1.4, 5])
    sigma = np.array([0.4, 1.2, 0.2])
    pi = np.array([0.3, 0.5, 0.2])
    mu0, sigma0, pi0 = 0.3, 0.5, 0.3 # Null distribution

    # Sample some data points from a mixture of gaussians
    c = np.random.choice(3, p=pi, size=N)
    y = np.random.normal(mu[c], sigma[c])
    density_truth = GmmDummy(pi, mu, sigma)

    # Estimate the null
    results = estimate_density(y)
    density_fit = results['dist']
    bins = results['bins']

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.hist(y, bins=30, density=True, color='gray', alpha=0.5, label='Observations')
    plt.plot(bins, density_truth.pdf(bins), color='black', label='Truth')
    plt.plot(bins, density_fit.pdf(bins), color='orange', label='Estimate')
    plt.legend(loc='upper left')
    plt.savefig('plots/predictive-recursion-fit.pdf', bbox_inches='tight')
    plt.close()

    plt.plot(results['sbins'], results['logm_grid'], lw=2)
    plt.xlabel('Kernel scale')
    plt.ylabel('PR marginal log-likelihood')
    plt.savefig('plots/predictive-recursion-logm.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    test_estimate_density()





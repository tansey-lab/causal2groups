import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from causal2groups.adadetect.procedure import AdaDetectERM
from causal2groups.utils import posterior_selection_fdr
from sklearn.datasets import fetch_openml 
from itertools import product

def sample_instance(outlr:np.ndarray, inlr:np.ndarray, n_in:int, n_out:int, n_null:int, seed:int):
    rng = np.random.default_rng(seed)
    x_in_and_null = rng.choice(inlr, size=(n_in+n_null), replace=False)
    x_in = x_in_and_null[:n_in]
    x_null = x_in_and_null[n_in:]
    
    x_out = rng.choice(outlr, size=n_out, replace=False)
    x = np.concatenate([x_out, x_in])
    outlier_mask = np.concatenate([np.ones(x_out.shape[0]), np.zeros(x_in.shape[0])])
    
    return(x, x_null, outlier_mask)

def fit_scoring_model(x:np.ndarray, xnull:np.ndarray, seed:int, n_estimators:int, max_depth:int):
    x_train = np.concatenate([xnull, x])
    y_train = np.concatenate([np.zeros(xnull.shape[0]), np.ones(x.shape[0])])
    scoring_fn = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, max_depth=max_depth)
    scoring_fn.fit(x_train, y_train)
    return(scoring_fn)

def cv_score(
        x:np.ndarray, 
        xnull:np.ndarray, 
        seed:int, 
        k:int=4, 
        n_estimators:int=200,
        eps=1e-4, 
        max_depth:int=5):
    
    rng = np.random.default_rng(seed)
    n_x = x.shape[0]
    n_xnull = xnull.shape[0]
    x_scores = np.zeros((n_estimators,n_x))
    xnull_scores = np.zeros((n_estimators,n_xnull))

    x_perm = rng.permutation(n_x)
    xnull_perm = rng.permutation(n_xnull)

    x_inds = np.array_split(x_perm, k)
    xnull_inds = np.array_split(xnull_perm, k)

    for i in range(k):
        x_test_idx = x_inds[i]
        xnull_test_idx = xnull_inds[i]

        x_train_idx = np.concatenate(x_inds[:i] + x_inds[(i+1):])
        xnull_train_idx = np.concatenate(xnull_inds[:i] + xnull_inds[(i+1):])

        scoring_fn:RandomForestClassifier = fit_scoring_model(x[x_train_idx], xnull[xnull_train_idx], seed, n_estimators, max_depth)

        n_0 = xnull_train_idx.shape[0]
        n_1 = x_train_idx.shape[0]
        correction_ratio = n_0/n_1

        p1 = np.stack([tree.predict_proba(x[x_test_idx])[:,1] for tree in scoring_fn.estimators_])
        x_scores[:,x_test_idx] = correction_ratio*p1/np.clip(1-p1, a_min=eps, a_max=None)

        
        p1 = np.stack([tree.predict_proba(xnull[xnull_test_idx])[:,1] for tree in scoring_fn.estimators_])
        xnull_scores[:,xnull_test_idx] = correction_ratio*p1/np.clip(1-p1, a_min=eps, a_max=None)
        
    return(x_scores, xnull_scores)


def np_c2g_eval(
        x_scores:np.ndarray, 
        xnull_scores:np.ndarray,
        outlier_mask:np.ndarray,
        nominal_levels:np.ndarray,
        quantile:float):
    
    pi = np.min(np.quantile(x_scores, quantile, axis=0))
    null_posterior_treated = (1.0 - pi)/np.quantile(x_scores, quantile, axis=0)
    null_posterior_untreated = (1.0 - pi)/np.quantile(xnull_scores, quantile, axis=0)
    null_posterior_probs = np.concatenate([null_posterior_treated, null_posterior_untreated])
    T = np.concatenate([np.ones(null_posterior_treated.shape[0]), np.zeros(null_posterior_untreated.shape[0])])
    H = np.concatenate([outlier_mask,  np.zeros(null_posterior_untreated.shape[0])])
    fdr_observed, power_observed = posterior_selection_fdr(null_posterior_probs, T, H, nominal_levels, empirical_control=True)
    return(fdr_observed, power_observed)

def adadetect(
        x:np.ndarray, 
        xnull:np.ndarray,
        outlier_mask:np.ndarray,
        nominal_levels:np.ndarray,
        seed:int, 
        n_estimators:int=200, 
        max_depth:int=5):
    
    proc = AdaDetectERM(scoring_fn= RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed), split_size=0.8)
    n_out = np.sum(outlier_mask)

    fdr_observed, power_observed = [], []
    for alpha in nominal_levels:
        rej = proc.apply(x, alpha, xnull) #gives the rejection set 
        if rej.shape[0]>0:
            power = outlier_mask[rej].sum()/n_out
            fdr = 1-outlier_mask[rej].mean()
        else:
            power = 0.0
            fdr = 0.0
        fdr_observed.append(fdr)
        power_observed.append(power)
    
    return(np.array(fdr_observed), np.array(power_observed))

def load_dataset(dataset_name:str):
    dataset = fetch_openml(name=dataset_name, version=1, as_frame=False)
    if dataset_name=='creditcard':
        X = dataset.data
        y = dataset.target.astype(np.float_)
        outlr, inlr = X[y==1], X[y==0]
    elif dataset_name=='shuttle':
        X = dataset.data
        y = dataset.target.astype(np.float_)
        outlr, inlr = X[y>1], X[y==1]
    elif dataset_name=='mammography':
        X = dataset.data
        y = dataset.target.astype(np.float_)
        outlr, inlr = X[y==1], X[y==-1]
    return(outlr, inlr)

if __name__ == '__main__':
    os.makedirs("./results/no_cov", exist_ok=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, help="Run index.")
    parser.add_argument('--n_jobs', type=int, help="Total number of jobs.")
    parser.add_argument('--n_seeds', type=int, help="Total number of seeds.")
    args = parser.parse_args()

    index:int = args.index
    n_jobs:int = args.n_jobs
    n_seeds:int = args.n_seeds

    seeds = np.arange(start=100, stop=(100+n_seeds))
    datasets = ['creditcard', 'shuttle', 'mammography']
    n_outs = [100]
    n_ins = [400]
    n_nulls = [500]

    settings = product(seeds, datasets, n_outs, n_ins, n_nulls)
    curr_settings = [x for i,x in enumerate(settings) if (i%n_jobs)==index]

    alphas = [0.01, 0.05, 0.1, 0.2]
    nominal_levels = np.linspace(0.0, 1.0, num=1000)
    quantiles = [0.35]
    n_estimators = 200
    max_depth = 5
    rows = []
    for seed, dataset_name, n_out, n_in, n_null in tqdm(curr_settings):
        outlr,inlr = load_dataset(dataset_name)
        x, x_null, outlier_mask = sample_instance(outlr, inlr, n_in, n_out, n_null, seed)

        ## np c2g
        x_scores, xnull_scores = cv_score(x, x_null, seed, n_estimators=n_estimators, max_depth=max_depth)

        for q in quantiles:
            fdr_observed, power_observed = np_c2g_eval(x_scores, xnull_scores, outlier_mask, nominal_levels, q)
            fdr_observed = np.interp(alphas, nominal_levels, fdr_observed)
            power_observed = np.interp(alphas, nominal_levels, power_observed)
            for alpha, fdr, power in zip(alphas, fdr_observed, power_observed):
                rows.append({"seed":seed, 
                             "dataset":dataset_name, 
                             "n_out":n_out, 
                             "n_in":n_in, 
                             "n_null":n_null, 
                             "method":"NP-C2G", 
                             "quantile":q, 
                             "nominal_fdr":alpha, 
                             "observed_fdr":fdr, 
                             "power":power})
        
        ## adadetect
        fdr_observed, power_observed = adadetect(x, x_null, outlier_mask, alphas, seed, n_estimators=n_estimators, max_depth=max_depth)
        for alpha, fdr, power in zip(alphas, fdr_observed, power_observed):
            rows.append({"seed":seed, 
                            "dataset":dataset_name, 
                            "n_out":n_out, 
                            "n_in":n_in, 
                            "n_null":n_null, 
                            "method":"AdaDetect", 
                            "nominal_fdr":alpha, 
                            "observed_fdr":fdr, 
                            "power":power})
    df = pd.DataFrame(rows)
    df.to_csv("./results/no_cov/{}.csv".format(index))


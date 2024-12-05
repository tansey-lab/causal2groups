import numpy as np


def posterior_selection(null_posterior_probs:np.ndarray, T:np.ndarray, fdr_levels:np.ndarray):
    treat_idx = np.where(T==1)[0]
    n_treated = treat_idx.shape[0]

    null_posterior_treated = null_posterior_probs[treat_idx]
    treat_order = np.argsort(null_posterior_treated)
    
    null_posterior_treated = null_posterior_treated[treat_order]
    treated_running_average = np.cumsum(null_posterior_treated)/np.arange(1,n_treated+1)
    selections = []
    for alpha in fdr_levels:
        mask = treated_running_average<=alpha
        selections.append(treat_idx[treat_order[mask]])
    return(selections)

def posterior_selection_empirical_control(null_posterior_probs:np.ndarray, T:np.ndarray, fdr_levels:np.ndarray):
    treat_idx = np.where(T==1)[0]
    n_treated = treat_idx.shape[0]

    null_posterior_treated = null_posterior_probs[treat_idx]
    treat_order = np.argsort(null_posterior_treated)
    
    null_posterior_treated = null_posterior_treated[treat_order]
    treated_running_average = np.cumsum(null_posterior_treated)/np.arange(1,n_treated+1)

    null_posterior_null = null_posterior_probs[T==0]
    null_order = np.argsort(null_posterior_null)
    n_null = null_order.shape[0]
    null_posterior_null = null_posterior_null[null_order]
    null_running_average = np.cumsum(null_posterior_null)/np.arange(1,n_null+1)

    ## For each fdr level, how many treated v.s. untreated selections were made?
    n_treat_selected = np.array([np.sum(treated_running_average<=alpha) for alpha in fdr_levels])
    n_null_selected = np.array([np.sum(null_running_average<=alpha) for alpha in fdr_levels])

    ## Conservative fdr estimate is # null selected/ # treated selected
    fdr_estimate = n_null_selected/np.clip(n_treat_selected, 1, np.inf)

    selections = []
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
        selections.append(treat_idx[treat_order[mask]])
    
    return(selections)

def posterior_selection_fdr(null_posterior_probs:np.ndarray, T:np.ndarray, H:np.ndarray, fdr_levels:np.ndarray, empirical_control:bool):
    if empirical_control:
        selections = posterior_selection_empirical_control(null_posterior_probs, T, fdr_levels)
    else:
        selections = posterior_selection(null_posterior_probs, T, fdr_levels)


    n_pos = np.sum((T==1)&(H==1))

    fdr_observed = np.zeros_like(fdr_levels)
    power_observed = np.zeros_like(fdr_levels)

    for i, sel_idx in enumerate(selections):
        num_sel = sel_idx.shape[0]
        num_sel_pos = np.sum(H[sel_idx])
        num_sel_neg = num_sel - num_sel_pos
        fdr_observed[i] = 0 if num_sel==0 else num_sel_neg/num_sel
        power_observed[i] = num_sel_pos/n_pos
    
    return(fdr_observed, power_observed)
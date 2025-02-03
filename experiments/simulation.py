import os, sys
import numpy as np
import pandas as pd
from causal2groups.simulated_data import AdditiveSimulatedData, NonadditiveSimulatedData, GDSCSemiSynthetic
from causal2groups.npc2g import KernelNonparametricCausal2G
from causal2groups.frequentist import KernelFrequentist
from causal2groups.addc2g import AdditiveCausal2G
from itertools import product
import subprocess
import argparse
from scipy.stats import false_discovery_control

def run_simulation(dir_name, N, tau, seed):
    print("N={}, tau={}, seed={}".format(N, tau, seed))
    ## Set seed
    np.random.seed(seed)

    X = pd.read_csv(os.path.join(dir_name, "X.csv")).values
    Y = pd.read_csv(os.path.join(dir_name, "Y.csv")).values.squeeze()
    T = pd.read_csv(os.path.join(dir_name, "T.csv")).values.squeeze()
    H = pd.read_csv(os.path.join(dir_name, "H.csv")).values.squeeze()

    P = X.shape[1]
    fdr_levels = np.linspace(0.0, 1.0, num=1000)
    if not os.path.isfile(os.path.join(dir_name, "nonadditive_causal2groups_full.csv")):
        ## Fit nonadditive causal2groups
        kernel_causal2groups = KernelNonparametricCausal2G(kernel_n_neighbors=[50, 100, 200], 
                                                        kernel_bandwidth_neighbor_fracs=np.logspace(-3,0, num=10), 
                                                        verbose=True)
        kernel_causal2groups.fit(X=X, Y=Y, T=T)
        
        ## Raw null probability scores
        raw_df = pd.DataFrame({"H":H[T==1], "q_value":kernel_causal2groups.null_posterior[T==1]})
        raw_df.to_csv(os.path.join(dir_name, "nonadditive_causal2groups_raw.csv"))

        full_df = pd.DataFrame({"H":H, "T":T, "q_value":kernel_causal2groups.null_posterior})
        full_df.to_csv(os.path.join(dir_name, "nonadditive_causal2groups_full.csv"))


        ## No empirical control
        obs_fdr, obs_pow = kernel_causal2groups.calculate_fdr(T=T, H=H, fdr_levels=fdr_levels, empirical_control=False)
        fdr_df = pd.DataFrame({"Nominal FDR":fdr_levels, "Observed FDR":obs_fdr, "Observed power":obs_pow, "N":N, "tau":tau, "seed":seed})
        fdr_df.to_csv(os.path.join(dir_name, "nonadditive_causal2groups.csv"))

        ## With empirical control
        obs_fdr, obs_pow = kernel_causal2groups.calculate_fdr(T=T, H=H, fdr_levels=fdr_levels, empirical_control=True)
        fdr_df = pd.DataFrame({"Nominal FDR":fdr_levels, "Observed FDR":obs_fdr, "Observed power":obs_pow, "N":N, "tau":tau, "seed":seed})
        fdr_df.to_csv(os.path.join(dir_name, "nonadditive_causal2groups_ec.csv"))

        ## Compute ITE
        ite_upper, ite_lower = kernel_causal2groups.predict_ite()
        ite_df = pd.DataFrame({"ITE upper bound":ite_upper, "ITE lower bound":ite_lower})
        ite_df.to_csv(os.path.join(dir_name, "nonadditive_causal2groups_ite.csv"))

    if not os.path.isfile(os.path.join(dir_name, "additive_causal2groups_full.csv")):
        ## Fit additive causal2groups
        add_causal2groups = AdditiveCausal2G(n_covariates=P, 
                                             rff_dims=100,
                                             kernel_n_bandwidths=6, 
                                             kernel_reg_params=np.logspace(-5, 5, num=50),
                                             seed=seed,
                                             verbose=True)
        add_causal2groups.fit(X=X, Y=Y, T=T)
        
        raw_df = pd.DataFrame({"H":H[T==1], "q_value":add_causal2groups.null_posterior[T==1]})
        raw_df.to_csv(os.path.join(dir_name, "additive_causal2groups_raw.csv"))

        full_df = pd.DataFrame({"H":H, "T":T, "q_value":add_causal2groups.null_posterior})
        full_df.to_csv(os.path.join(dir_name, "additive_causal2groups_full.csv"))

        ## No empirical control
        obs_fdr, obs_pow = add_causal2groups.calculate_fdr(T=T, H=H, fdr_levels=fdr_levels, empirical_control=False)
        fdr_df = pd.DataFrame({"Nominal FDR":fdr_levels, "Observed FDR":obs_fdr, "Observed power":obs_pow, "N":N, "tau":tau, "seed":seed})
        fdr_df.to_csv(os.path.join(dir_name, "additive_causal2groups.csv"))

        ## With empirical control
        obs_fdr, obs_pow = add_causal2groups.calculate_fdr(T=T, H=H, fdr_levels=fdr_levels, empirical_control=True)
        fdr_df = pd.DataFrame({"Nominal FDR":fdr_levels, "Observed FDR":obs_fdr, "Observed power":obs_pow, "N":N, "tau":tau, "seed":seed})
        fdr_df.to_csv(os.path.join(dir_name, "additive_causal2groups_ec.csv"))

        ## Compute ITE
        ite_df = pd.DataFrame({"ITE":add_causal2groups.predict_ite()})
        ite_df.to_csv(os.path.join(dir_name, "additive_causal2groups_ite.csv"))

    if not os.path.isfile(os.path.join(dir_name, "frequentist_raw.csv")):
        ## Fit frequentist model
        kernel_freq = KernelFrequentist(kernel_n_neighbors=[50, 100, 200], 
                                        kernel_bandwidth_neighbor_fracs=np.logspace(-3,0, num=10))
        kernel_freq.fit(X=X, Y=Y, T=T)

        obs_fdr, obs_pow = kernel_freq.calculate_fdr(T=T, H=H, fdr_levels=fdr_levels)
        fdr_df = pd.DataFrame({"Nominal FDR":fdr_levels, "Observed FDR":obs_fdr, "Observed power":obs_pow, "N":N, "tau":tau, "seed":seed})
        fdr_df.to_csv(os.path.join(dir_name, "frequentist.csv"))

        raw_df = pd.DataFrame({"H":H[T==1], "q_value":false_discovery_control(kernel_freq.null_density_upper[T==1])})
        raw_df.to_csv(os.path.join(dir_name, "frequentist_raw.csv"))

    if not os.path.isfile(os.path.join(dir_name, "bart.csv")):
        ## Run BART
        subprocess.call(["Rscript", "--vanilla", "R/bart.R", dir_name])

    if not os.path.isfile(os.path.join(dir_name, "causal_forest.csv")):
        ## Run causal forest
        subprocess.call(["Rscript", "--vanilla", "R/causal_forests.R", dir_name])

    if not os.path.isfile(os.path.join(dir_name, "FDRreg.csv")):
        ## Run FDRreg
        subprocess.call(["Rscript", "--vanilla", "R/FDRreg.R", dir_name])


def job_complete(setting, setup):
    if setting in ['additive', 'nonadditive']:
        N, tau, seed = setup
        dir_name = "results/{}/N_{}_tau{}_seed_{}".format(setting, N, tau, seed)
    else:
        seed = setup
        dir_name = "results/nutlin/pca_seed_{}".format(seed)
    
    if not os.path.isdir(dir_name):
        return False
    
    fnames = ["nonadditive_causal2groups_full.csv", "additive_causal2groups_full.csv", 
              "frequentist_raw.csv", "bart.csv", "causal_forest.csv", "FDRreg.csv"]

    return(all([os.path.isfile(os.path.join(dir_name, x)) for x in fnames]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--setting', type=str, default="additive")
    args = parser.parse_args()

    n_workers:int = args.n_workers
    worker_id:int = args.worker_id
    setting:str = args.setting
    if worker_id >= n_workers:
        sys.exit()

    os.makedirs("results", exist_ok=True)

    
    ## Settings
    if setting in ['additive', 'nonadditive']:
        taus = [1, 3, 5]
        Ns = [1000, 10000]
        seeds = np.arange(100, 150)
        setups = list(product(Ns, taus, seeds))
    else:
        seeds = np.arange(100, 150)
        setups = seeds
        features_df = pd.read_csv("./data/nutlin/all_features.csv", index_col=0)
        outcomes_df = pd.read_csv('./data/nutlin/all_outcomes.csv')
        drug_df = pd.read_csv('./data/nutlin/gdsc_drug_details.csv')


    ## Assign each worker to its corresponding setting
    remaining_setups = [setup for setup in setups if not job_complete(setting, setup)]
    setup_assignment = np.array_split(remaining_setups, n_workers)
    curr_setups = setup_assignment[worker_id]


    for setup in curr_setups:
        if setting in ['additive', 'nonadditive']:
            N, tau, seed = setup
            dir_name = "results/{}/N_{}_tau{}_seed_{}".format(setting, N, tau, seed)
            os.makedirs(dir_name, exist_ok=True)
            if setting == 'additive':
                sim_data = AdditiveSimulatedData(P=10, tau=tau, seed=seed)
            else:
                sim_data = NonadditiveSimulatedData(P=10, tau=tau, seed=seed)

            X, Y, T, H, H_prob = sim_data.generate_data(N)
        else:
            seed = setup
            N = 0
            tau = 0
            dir_name = "results/nutlin/pca_seed_{}".format(seed)
            sim_data = GDSCSemiSynthetic(features_df=features_df,
                                 outcomes_df=outcomes_df,
                                 drug_df=drug_df, 
                                 drug='Nutlin-3a (-)',
                                 mutations=['TP53'], 
                                 conditions=[0], 
                                 seed=seed)

            X, Y, T, H = sim_data.generate_data(pca=True)
        
        ## Write out the directory
        os.makedirs(dir_name, exist_ok=True)
        
        ## Boolean variables -> 0, 1 integer
        T = T.astype(int)
        H = H.astype(int)
        
        ## Save the data out
        for a,a_name in [(X,"X.csv"), (Y, "Y.csv"), (T, "T.csv"), (H, "H.csv")]:
            df = pd.DataFrame(data=a)
            df.to_csv(os.path.join(dir_name, a_name), index=False)

        ## Run the simulation
        run_simulation(dir_name, N, tau, seed)


















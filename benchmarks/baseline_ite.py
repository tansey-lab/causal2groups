import os
import numpy as np
import pandas as pd
from causal2groups.simulated_data import GDSCSemiSynthetic
import subprocess

def run_ite(dir_name, seed):
    ## Set seed
    np.random.seed(seed)

    subprocess.call(["Rscript", "--vanilla", "R/bart.R", dir_name])

    subprocess.call(["Rscript", "--vanilla", "R/causal_forests.R", dir_name])

if __name__ == '__main__':
    features_df = pd.read_csv("./data/all_features.csv", index_col=0)
    outcomes_df = pd.read_csv('./data/all_outcomes.csv')
    drug_df = pd.read_csv('./data/gdsc_drug_details.csv')

    seed = 100
    dir_name = "results/nutlin/baseline"
    sim_data = GDSCSemiSynthetic(features_df=features_df,
                            outcomes_df=outcomes_df,
                            drug_df=drug_df, 
                            drug='Nutlin-3a (-)',
                            mutations=['TP53'], 
                            conditions=[0], 
                            seed=seed)

    X, Y, _, H = sim_data.generate_data(pca=True)
        
    ## Write out the directory
    os.makedirs(dir_name, exist_ok=True)
        
    ## Boolean variables -> 0, 1 integer
    H = H.astype(int)
    
    ## Save the data out, writing out H as T this time
    for a,a_name in [(X, "X.csv"), (Y, "Y.csv"), (H, "T.csv"), (H, "H.csv")]: 
        df = pd.DataFrame(data=a)
        df.to_csv(os.path.join(dir_name, a_name), index=False)

    ## Run the simulation
    run_ite(dir_name, seed)


















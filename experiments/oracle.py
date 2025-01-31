import numpy as np
import pandas as pd
import os
from causal2groups.simulated_data import AdditiveSimulatedData, NonadditiveSimulatedData
from scipy.integrate import simpson
from tqdm import tqdm

dfs = []
base_folder = "./results"
for setting in ["additive", "nonadditive"]:
    folders = [x for x in os.listdir(os.path.join(base_folder, setting)) if x.startswith("N_")]
    for folder_name in tqdm(folders):
        _,N,tau,_,seed = folder_name.split('_')
        tau = float(tau[3:])
        N = int(N)
        seed = int(seed)
        if setting == 'additive':
            sim_data = AdditiveSimulatedData(P=10, tau=tau, seed=seed)
        else:
            sim_data = NonadditiveSimulatedData(P=10, tau=tau, seed=seed)

        ## Generate data
        X, Y, T, H, H_prob = sim_data.generate_data(N)
        y_grid = np.linspace( (Y.min()-3), (Y.max()+3), num=300)
        nulls = sim_data.conditional_null_density(X, y_grid)
        treats = sim_data.conditional_treat_density(X, y_grid)

        min_null = np.min(nulls[nulls>0])
        nulls = np.clip(nulls, a_min=min_null, a_max=None)

        min_treat = np.min(treats[treats>0])
        treats = np.clip(treats, a_min=min_treat, a_max=None)
        
        ratio = np.exp(np.log(treats) - np.log(nulls))
        pi_star = 1 - np.nanmin(ratio, axis=1)
        alts_star = (1/pi_star[:,np.newaxis])*(treats - nulls) + nulls

        ## Calculate ITEs
        null_mu = simpson(x=y_grid, y=(y_grid*nulls))
        alt_mu = simpson(x=y_grid, y=(y_grid*alts_star))
        upper_ite = alt_mu-null_mu
        treat_mu = simpson(x=y_grid, y=(y_grid*treats))
        lower_ite = treat_mu-null_mu

        ## Calculate ground truth conservative values
        idx_upper = np.searchsorted(y_grid, Y)
        idx_lower = np.clip(idx_upper-1, a_min=0, a_max=None)
        Y_upper = y_grid[idx_upper]
        Y_lower = y_grid[idx_lower]
        alpha = (Y - Y_lower)/(Y_upper-Y_lower)
        null_pdf = (1-alpha)*nulls[np.arange(N), idx_lower] + alpha * nulls[np.arange(N), idx_upper]
        treat_pdf = (1-alpha)*treats[np.arange(N), idx_lower] + alpha * treats[np.arange(N), idx_upper]
        p_null = (1-pi_star)*null_pdf/treat_pdf

        ite = sim_data.ite(X)
        dfs.append(pd.DataFrame({"data_index":np.arange(N),
                                 "T":T,
                                 "H":H,
                                 "ITE lower bound":lower_ite, 
                                 "ITE upper bound":upper_ite, 
                                 "P null":p_null,
                                 "Ground-truth ITE":ite, 
                                 "N":N, 
                                 "tau":tau, 
                                 "seed":seed, 
                                 "setting":setting}))
        
df = pd.concat(dfs, ignore_index=True)

df.to_csv("./results/oracle.csv", index=0)


ref_df = df[["setting", "seed", "tau", "N", "P null", "T", "H"]].copy()
ref_df = ref_df[ref_df["T"]==1]
ref_df = ref_df.sort_values(by=["setting", "seed", "tau", "N", "P null"]).reset_index(drop=True)
ref_df['Cumulative p_null'] = ref_df.groupby(["setting", "seed", "tau", "N"])['P null'].cumsum()
ref_df['temp'] = 1.0
ref_df['q_value'] = ref_df['Cumulative p_null']/ref_df.groupby(["setting", "seed", "tau", "N"])['temp'].cumsum()
fdr_levels = np.linspace(0.0, 1.0, num=1000).round(3)
groupings = ref_df.groupby(["setting", "N", "seed", "tau"]).groups
dfs = []
for (setting, seed, tau, N), idx in tqdm(groupings.items()):
    sub_df = ref_df.iloc[idx]
    qvals = sub_df['q_value'].values
    H_treated = sub_df['H'].values
    n_pos = np.sum(H_treated)
    fdr_observed = np.zeros_like(fdr_levels)
    power_observed = np.zeros_like(fdr_levels)
    for i, alpha in enumerate(fdr_levels):
        mask = qvals<=alpha
        num_sel = np.sum(mask)
        num_neg = num_sel - np.sum(H_treated[mask])
        fdr_observed[i] = 0 if num_sel==0 else num_neg/num_sel
        power_observed[i] = np.sum(H_treated[mask])/n_pos
    dfs.append(pd.DataFrame({"Nominal FDR":fdr_levels, "Observed FDR":fdr_observed, "Observed power":power_observed, "N":N, "tau":tau, "seed":seed, "setting":setting}))
df = pd.concat(dfs, ignore_index=True)
df.to_csv("./results/oracle_individual_fdr.csv", index=0)


mean_df = df.drop(columns="seed").groupby(["setting", "N", "tau", "Nominal FDR"]).mean().reset_index()
std_df = df.drop(columns="seed").groupby(["setting","N", "tau", "Nominal FDR"]).std().reset_index()
nobs = df.drop(columns="seed").groupby(["setting","N", "tau", "Nominal FDR"]).size().reset_index()
res_df = mean_df.merge(std_df, on=["setting","N", "tau", "Nominal FDR"], suffixes=('_mean', '_stdv'))
res_df = res_df.merge(nobs, on=["setting", "N", "tau", "Nominal FDR"])
res_df.rename(columns={0:"count"}, inplace=True)
res_df['Observed FDR_CI'] = 1.96*res_df['Observed FDR_stdv']/np.sqrt(res_df['count'])
res_df['Observed FDR_STDERR'] = res_df['Observed FDR_stdv']/np.sqrt(res_df['count'])
res_df['Observed power_CI'] = 1.96*res_df['Observed power_stdv']/np.sqrt(res_df['count'])

res_df['Observed FDR_mean'] = res_df['Observed FDR_mean'].round(2)
res_df['Observed FDR_CI'] = res_df['Observed FDR_CI'].round(2)

## Calculate valid power
m_df = df.merge(res_df, on=["setting", "N", "tau", "Nominal FDR"])
m_df['Valid'] = (m_df['Observed FDR_mean']-m_df['Observed FDR_CI'])<=m_df['Nominal FDR']
m_df = m_df[["setting", "N", "tau", "seed", "Nominal FDR", "Observed power", "Valid"]]
m_df['Intermediate power'] = np.where(m_df['Valid'], m_df['Observed power'], 0)
m_df['Valid power'] = m_df['Intermediate power']
mean_df = m_df.drop(columns="seed").groupby(["setting", "N", "tau","Nominal FDR"]).mean().reset_index()
std_df = m_df.drop(columns="seed").groupby(["setting", "N", "tau", "Nominal FDR"]).std().reset_index()
nobs = m_df.drop(columns="seed").groupby(["setting", "N", "tau", "Nominal FDR"]).size().reset_index()

mres_df = mean_df.merge(std_df, on=["setting", "N", "tau", "Nominal FDR"], suffixes=('_mean', '_stdv'))
mres_df = mres_df.merge(nobs, on=["setting", "N", "tau", "Nominal FDR"])
mres_df.rename(columns={0:"count"}, inplace=True)
mres_df['Valid power_CI'] = 1.96*mres_df['Valid power_stdv']/np.sqrt(mres_df['count'])
mres_df = mres_df[["setting", "N", "tau", "Nominal FDR", 'Valid power_mean', 'Valid power_CI']]

mres_df["Valid power_mean"] = mres_df["Valid power_mean"].round(2)
mres_df["Valid power_CI"] = mres_df["Valid power_CI"].round(2)

fdf = res_df.merge(mres_df, on=["setting", "N", "tau", "Nominal FDR"])
fdf.rename(columns={"setting":"Setting"}, inplace=True)
fdf.to_csv("./results/oracle_compressed_fdr.csv", index=0)
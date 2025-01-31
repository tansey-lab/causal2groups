import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve
from scipy.integrate import simpson
from causal2groups.simulated_data import AdditiveSimulatedData, NonadditiveSimulatedData

all_methods = ["additive_causal2groups", 
               "causal_forest", 
               'bart', 
               "additive_causal2groups_ec", 
               "nonadditive_causal2groups", 
               "nonadditive_causal2groups_ec", 
               "causal2groups", 
               "causal2groups_ec", 
               "FDRreg", 
               'frequentist']

abbrev2full = {"causal_forest":"Causal forest", 
               "bart":"BART", 
               "additive_causal2groups":"Add-C2G", 
               "additive_causal2groups_ec":"Add-C2G-EC", 
               "nonadditive_causal2groups":"NP-C2G", 
               "nonadditive_causal2groups_ec":"NP-C2G-EC", 
               "FDRreg":"FDRreg", 
               "frequentist":"Frequentist"}


def load_raw(folder):
    folder_names = [x for x in os.listdir(folder) if x.startswith('N_') or x.startswith('pca_')]
    raw_dfs = []
    for folder_name in folder_names:
        if folder_name.startswith('N_'):
            _,N,tau,_,seed = folder_name.split('_')
            tau = tau[3:]
        else:
            *_,seed = folder_name.split('_')
            N = 0 
            tau = 0
        for model in all_methods:
            fname = os.path.join(folder, folder_name, model+"_raw.csv")
            if os.path.isfile(fname):
                raw_df = pd.read_csv(fname, index_col=0)
                raw_df.rename(columns={x:' '.join(x.split('.')) for x in raw_df.columns}, inplace=True)
                raw_df['seed'] = int(seed)
                raw_df['N'] = int(N)
                raw_df['tau'] = float(tau)                    
                raw_df['method'] = model
                raw_dfs.append(raw_df)
    
    raw_df = pd.concat(raw_dfs, ignore_index=True)
    raw_df.fillna(0, inplace=True)
    raw_df.replace(abbrev2full, inplace=True)
    return(raw_df)


def load_ite(folder):
    folder_names = [x for x in os.listdir(folder) if x.startswith('N_') or x.startswith('pca_')]
    ite_dfs = []
    for folder_name in folder_names:
        if folder_name.startswith('N_'):
            _,N,tau,_,seed = folder_name.split('_')
            tau = float(tau[3:])
            seed = int(seed)
            N = int(N)
        else:
            *_,seed = folder_name.split('_')
            N = 0 
            tau = 0
            seed = int(seed)

        for model in all_methods:
            fname = os.path.join(folder, folder_name, model+"_ite.csv")
            if os.path.isfile(fname):
                ite_df = pd.read_csv(fname, index_col=0)                  
                ite_df['method'] = model
                ite_df.reset_index(names="data_index", inplace=True)
                if model in ('bart', 'causal_forest'):
                    ite_df['data_index'] = ite_df['data_index'] - 1
                ite_df['seed'] = seed
                ite_df['N'] = N
                ite_df['tau'] = tau
                ite_dfs.append(ite_df)
        if ('additive' in folder) and ('nonadditive' not in folder):
            sim_data = AdditiveSimulatedData(P=10, tau=tau, seed=seed)
            X, Y, T, H, H_prob = sim_data.generate_data(N=N)
            ite = sim_data.ite(X)
        elif 'nonadditive' in folder:
            sim_data = NonadditiveSimulatedData(P=10, tau=tau, seed=seed)
            X, Y, T, H, H_prob = sim_data.generate_data(N=N)
            ite = sim_data.ite(X)
        ite_df = pd.DataFrame({"ITE":ite, "method":"ground_truth", "seed":seed, "N":N,"tau":tau})
        ite_df.reset_index(names="data_index", inplace=True)
        ite_dfs.append(ite_df)

    ite_df = pd.concat(ite_dfs, ignore_index=True)
    pivot_df = ite_df.pivot(index=["seed", "N", "tau", "data_index"], columns="method", values="ITE").reset_index()
    
    ## Correlation of ITEs
    corr_df = pivot_df[["N", "tau", "seed", "additive_causal2groups", "nonadditive_causal2groups", "bart", "causal_forest", "ground_truth"]].groupby(["N", "tau", "seed"]).corr().reset_index()
    corr_df = corr_df[["N", "tau", "seed", "method", "ground_truth"]].rename(columns={"ground_truth":"ITE correlation"})

    mean_df = corr_df.drop(columns="seed").groupby(["N", "tau", "method"]).mean().reset_index()
    std_df = corr_df.drop(columns="seed").groupby(["N", "tau", "method"]).std().reset_index()
    nobs = corr_df.drop(columns="seed").groupby(["N", "tau", "method"]).size().reset_index()
    res_df = mean_df.merge(std_df, on=["N", "tau", "method"], suffixes=('_mean', '_stdv'))
    res_df = res_df.merge(nobs, on=["N", "tau", "method"])
    res_df.rename(columns={0:"count"}, inplace=True)
    res_df['ITE correlation_CI'] = 1.96*res_df['ITE correlation_stdv']/np.sqrt(res_df['count'])

    corr_df = res_df[res_df['method']!='ground_truth'].reset_index(drop=True)
    corr_df.replace(abbrev2full, inplace=True)

    ## ATE bias
    ate_df = pivot_df.drop(columns="data_index").groupby(["seed", "N", "tau"]).mean().reset_index()
    for method in ["additive_causal2groups", "nonadditive_causal2groups", "bart", "causal_forest"]:
        ate_df[method+'_ate_bias'] = ate_df[method] - ate_df['ground_truth']
    ate_df = ate_df[['N', 'tau'] + [x for x in ate_df.columns if x.endswith('_ate_bias')]]
    ate_df = ate_df.melt(id_vars=['N', 'tau'], value_vars=[x for x in ate_df.columns if x.endswith('_ate_bias')], var_name='method', value_name='ATE bias')
    ate_df = ate_df.replace({(method+'_ate_bias'):method for method in ["additive_causal2groups", "nonadditive_causal2groups", "bart", "causal_forest"]})

    mean_df = ate_df.groupby(["N", "tau", "method"]).mean().reset_index()
    std_df = ate_df.groupby(["N", "tau", "method"]).std().reset_index()
    nobs = ate_df.groupby(["N", "tau", "method"]).size().reset_index()
    res_df = mean_df.merge(std_df, on=["N", "tau", "method"], suffixes=('_mean', '_stdv'))
    res_df = res_df.merge(nobs, on=["N", "tau", "method"])
    res_df.rename(columns={0:"count"}, inplace=True)
    res_df['ATE bias_CI'] = 1.96*res_df['ATE bias_stdv']/np.sqrt(res_df['count'])

    ate_df = res_df[res_df['method']!='ground_truth'].reset_index(drop=True)
    ate_df.replace(abbrev2full, inplace=True)

    return(ate_df.merge(corr_df, on=["N", "tau", "method"]))


def load_roc(folder):
    raw_df = load_raw(folder)
    num = 1000
    grid = np.concatenate([ np.linspace(0,0.5/num, num=num) , np.linspace(1./num, (1-(1/num)), num=num), np.linspace((1-(0.5/num)),1, num=num) ])
        
    df_groups = raw_df.groupby(['N', 'tau', 'seed','method']).indices
    roc_dfs = []
    auc_dfs = []
    for (N, tau, seed, method), idx in (df_groups.items()):
        sub_df = raw_df.loc[idx]
        fpr, tpr, _ = roc_curve(y_true=sub_df['H'].values, y_score=(1-sub_df['q_value'].values))
        s = np.where( fpr[1:]>fpr[:-1])[0]
        condensed_fpr = np.concatenate([[0],fpr[s], [1]])
        condensed_tpr = np.concatenate([[0],tpr[s], [1]])
        vals = np.interp(x=grid, xp=condensed_fpr, fp=condensed_tpr)
        
        temp_df = pd.DataFrame({'N':N, 'tau':tau, 'seed':seed, 'method':method,
                                'False positive rate':np.concatenate([[0],grid, [1]]), 
                                'True positive rate':np.concatenate([[0],vals, [1]])})

        roc_dfs.append(temp_df)


        auc = simpson(x=temp_df["False positive rate"], y=temp_df["True positive rate"])
        auc_dfs.append(pd.DataFrame({'N':N, 'tau':tau, 'seed':seed, 'method':method, 'AUC':[auc]}))

    roc_df = pd.concat(roc_dfs, ignore_index=True)
    auc_df = pd.concat(auc_dfs, ignore_index=True)

    ## Confidence intervals for roc curves
    mean_df = roc_df.drop(columns="seed").groupby(["N", "tau", "method", "False positive rate"]).mean().reset_index()
    std_df = roc_df.drop(columns="seed").groupby(["N", "tau", "method", "False positive rate"]).std().reset_index()
    nobs = roc_df.drop(columns="seed").groupby(["N", "tau", "method", "False positive rate"]).size().reset_index()
    res_df = mean_df.merge(std_df, on=["N", "tau", "method", "False positive rate"], suffixes=('_mean', '_stdv'))
    res_df = res_df.merge(nobs, on=["N", "tau", "method", "False positive rate"])
    res_df.rename(columns={0:"count"}, inplace=True)
    res_df['True positive rate_CI'] = 1.96*res_df['True positive rate_stdv']/np.sqrt(res_df['count'])
    roc_df = res_df.copy()
    
    ## Confidence intervals for AUCs
    mean_df = auc_df.drop(columns="seed").groupby(["N", "tau", "method"]).mean().reset_index()
    std_df = auc_df.drop(columns="seed").groupby(["N", "tau", "method"]).std().reset_index()
    nobs = auc_df.drop(columns="seed").groupby(["N", "tau", "method"]).size().reset_index()
    res_df = mean_df.merge(std_df, on=["N", "tau", "method"], suffixes=('_mean', '_stdv'))
    res_df = res_df.merge(nobs, on=["N", "tau", "method"])
    res_df.rename(columns={0:"count"}, inplace=True)
    res_df['AUC_CI'] = 1.96*res_df['AUC_stdv']/np.sqrt(res_df['count'])

    res_df['AUC_mean'] = res_df['AUC_mean'].round(2)
    res_df['AUC_CI'] = res_df['AUC_CI'].round(2)

    auc_df = res_df

    return(roc_df, auc_df)

def load_fdr(folder):
    folder_names = [x for x in os.listdir(folder) if x.startswith('N_') or x.startswith('pca_')]
    dfs = []
    for folder_name in folder_names:
        if folder_name.startswith('N_'):
            _,N,tau,_,seed = folder_name.split('_')
            tau = tau[3:]
        else:
            *_,seed = folder_name.split('_')
            N = 0 
            tau = 0
        for model in all_methods:
            fname = os.path.join(folder, folder_name, model+".csv")
            if os.path.isfile(fname):
                df = pd.read_csv(fname, index_col=0)
                df.rename(columns={x:' '.join(x.split('.')) for x in df.columns}, inplace=True)
                df.rename(columns={"False discovery rate":"Observed FDR"}, inplace=True)
                df.rename(columns={"Fraction of discoveries made":"Observed power"}, inplace=True)
                df['seed'] = int(seed)
                df['N'] = int(N)
                df['tau'] = float(tau)
                df['method'] = model
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.fillna(0, inplace=True)
    df = df[["Nominal FDR", "Observed FDR", "Observed power", "N", "tau", "seed", "method"]]
    df.replace(abbrev2full, inplace=True)
    df['Nominal FDR'] = df['Nominal FDR'].round(3)

    ## Calculate standard FDR + Power
    mean_df = df.drop(columns="seed").groupby(["N", "tau", "method", "Nominal FDR"]).mean().reset_index()
    std_df = df.drop(columns="seed").groupby(["N", "tau", "method", "Nominal FDR"]).std().reset_index()
    nobs = df.drop(columns="seed").groupby(["N", "tau", "method", "Nominal FDR"]).size().reset_index()
    res_df = mean_df.merge(std_df, on=["N", "tau", "method", "Nominal FDR"], suffixes=('_mean', '_stdv'))
    res_df = res_df.merge(nobs, on=["N", "tau", "method", "Nominal FDR"])
    res_df.rename(columns={0:"count"}, inplace=True)
    res_df['Observed FDR_CI'] = 1.96*res_df['Observed FDR_stdv']/np.sqrt(res_df['count'])
    res_df['Observed FDR_STDERR'] = res_df['Observed FDR_stdv']/np.sqrt(res_df['count'])
    res_df['Observed power_CI'] = 1.96*res_df['Observed power_stdv']/np.sqrt(res_df['count'])

    res_df['Observed FDR_mean'] = res_df['Observed FDR_mean'].round(2)
    res_df['Observed FDR_CI'] = res_df['Observed FDR_CI'].round(2)

    ## Calculate valid power
    m_df = df.merge(res_df, on=["N", "tau", "method", "Nominal FDR"])
    m_df['Valid'] = (m_df['Observed FDR_mean']-m_df['Observed FDR_CI'])<=m_df['Nominal FDR']
    m_df = m_df[["N", "tau", "method", "seed", "Nominal FDR", "Observed power", "Valid"]]
    m_df['Intermediate power'] = np.where(m_df['Valid'], m_df['Observed power'], 0)
    m_df['Valid power'] = m_df['Intermediate power']
    # m_df['Valid power'] = m_df.groupby(["N", "tau", "method", "seed"])['Intermediate power'].cummax()
    mean_df = m_df.drop(columns="seed").groupby(["N", "tau", "method", "Nominal FDR"]).mean().reset_index()
    std_df = m_df.drop(columns="seed").groupby(["N", "tau", "method", "Nominal FDR"]).std().reset_index()
    nobs = m_df.drop(columns="seed").groupby(["N", "tau", "method", "Nominal FDR"]).size().reset_index()

    mres_df = mean_df.merge(std_df, on=["N", "tau", "method", "Nominal FDR"], suffixes=('_mean', '_stdv'))
    mres_df = mres_df.merge(nobs, on=["N", "tau", "method", "Nominal FDR"])
    mres_df.rename(columns={0:"count"}, inplace=True)
    mres_df['Valid power_CI'] = 1.96*mres_df['Valid power_stdv']/np.sqrt(mres_df['count'])
    mres_df = mres_df[["N", "tau", "method", "Nominal FDR", 'Valid power_mean', 'Valid power_CI']]

    mres_df["Valid power_mean"] = mres_df["Valid power_mean"].round(2)
    mres_df["Valid power_CI"] = mres_df["Valid power_CI"].round(2)

    fdf = res_df.merge(mres_df, on=["N", "tau", "method", "Nominal FDR"])
    return(fdf)


class ResultsInterpreter:
    def __init__(self, result_folder:str, keep_no_ec:bool=False):
        self.result_folder = result_folder
        self.keep_no_ec = keep_no_ec

        self.roc_df = None
        self.auc_df = None
        self.fdr_df = None
        self.ite_df = None

    def get_df(self, metric):
        if metric=="roc":
            if self.roc_df is None:
                self.load_roc()
            return self.roc_df.copy()
        elif metric in ["fdr", "power"]:
            if self.fdr_df is None:
                self.load_fdr()
            return self.fdr_df.copy()
        elif metric in ['ate', 'corr']:
            if self.ite_df is None:
                self.load_ite()
            return self.ite_df.copy()
        else:
            if self.auc_df is None:
                self.load_roc()
            return self.auc_df.copy()
    
    def load_fdr(self):
        fdr_dfs = []
        for setting in ['additive', 'nonadditive', 'nutlin']:
            folder = os.path.join(self.result_folder, setting)
            fdr_df = load_fdr(folder)
            fdr_df['Setting'] = setting
            fdr_dfs.append(fdr_df)
        self.fdr_df = pd.concat(fdr_dfs, ignore_index=True)

        if not self.keep_no_ec:
            self.fdr_df = self.fdr_df[~self.fdr_df['method'].isin(['Add-C2G', "NP-C2G"])].reset_index(drop=True)
            self.fdr_df.replace('Add-C2G-EC', 'Add-C2G', inplace=True)
            self.fdr_df.replace('NP-C2G-EC', 'NP-C2G', inplace=True)

        oracle_df = pd.read_csv(os.path.join(self.result_folder, "oracle_compressed_fdr.csv"))
        oracle_df["method"] = "NP-Oracle"
        self.fdr_df = pd.concat([self.fdr_df,oracle_df], ignore_index=True)

    def load_roc(self):
        roc_dfs = []
        auc_dfs = []
        for setting in ['additive', 'nonadditive', 'nutlin']:
            folder = os.path.join(self.result_folder, setting)
            roc_df, auc_df = load_roc(folder)
            roc_df['Setting'] = setting
            auc_df['Setting'] = setting
            roc_dfs.append(roc_df)
            auc_dfs.append(auc_df)
        
        self.roc_df = pd.concat(roc_dfs, ignore_index=True)
        self.auc_df = pd.concat(auc_dfs, ignore_index=True)

    def load_ite(self):
        ite_dfs = []
        for setting in ['additive', 'nonadditive']:
            folder = os.path.join(self.result_folder, setting)
            ite_df = load_ite(folder)
            ite_df['Setting'] = setting
            ite_dfs.append(ite_df)
        
        self.ite_df = pd.concat(ite_dfs, ignore_index=True)

    def fdr_lookup(self, val):
        fdr_df = self.get_df('fdr')
        sub_df = fdr_df.loc[( fdr_df['Nominal FDR']==val), ['N', 'tau', 'method', 'Setting', 'Observed FDR_mean', 'Observed FDR_CI']].copy()
        sub_df['Observed FDR_mean'] = sub_df['Observed FDR_mean']
        sub_df['Observed FDR_CI'] = sub_df['Observed FDR_CI']
        sub_df['Observed FDR_bold'] = (sub_df['Observed FDR_mean'] - sub_df['Observed FDR_CI']) <= val
        return(sub_df.set_index(['N', 'tau', 'method', 'Setting']).to_dict())

    def power_lookup(self, val):
        fdr_df = self.get_df('fdr')
        sub_df = fdr_df.loc[(fdr_df['Nominal FDR']==val), ['N', 'tau', 'method', 'Setting', 'Valid power_mean', 'Valid power_CI']].copy()
        sub_df['Valid power_mean'] = sub_df['Valid power_mean']
        sub_df['Valid power_CI'] = sub_df['Valid power_CI']
        sub_df['Valid power_low'] = sub_df['Valid power_mean'] - sub_df['Valid power_CI']
        sub_df['Valid power_high'] = sub_df['Valid power_mean'] + sub_df['Valid power_CI']
        sub_df = sub_df.merge(sub_df.groupby(["N", "tau", "Setting"])['Valid power_low'].max().reset_index().rename(columns={"Valid power_low":"thresh"}))
        sub_df['Valid power_bold'] = sub_df['Valid power_high'] >= sub_df['thresh']
        return(sub_df.set_index(['N', 'tau', 'method', 'Setting']).to_dict())
    
    def auc_lookup(self):
        sub_df = self.get_df('auc')
        sub_df['AUC_mean'] = sub_df['AUC_mean']
        sub_df['AUC_CI'] = sub_df['AUC_CI']
        sub_df['AUC_low'] = sub_df['AUC_mean'] - sub_df['AUC_CI']
        sub_df['AUC_high'] = sub_df['AUC_mean'] + sub_df['AUC_CI']
        sub_df = sub_df.merge(sub_df.groupby(["N", "tau", "Setting"])['AUC_low'].max().reset_index().rename(columns={"AUC_low":"thresh"}))
        sub_df['AUC_bold'] = sub_df['AUC_high'] >= sub_df['thresh']
        return(sub_df.set_index(['N', 'tau', 'method', 'Setting']).to_dict())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.metrics import roc_curve
from matplotlib import colormaps

lw = 4
all_methods = ["additive_causal2groups", 
               "causal_forest", 
               'bart', 
               "additive_causal2groups_ec", 
               "nonadditive_causal2groups", 
               "nonadditive_causal2groups_ec", 
               "causal2groups", 
               "causal2groups_ec", 
               "FDRreg"]

def get_ci(sub_df, x_label, y_label):
    sub_df = sub_df[[x_label, y_label]]
    std_df = sub_df.groupby([x_label]).std().reset_index()
    mean_df = sub_df.groupby([x_label]).mean().reset_index()
    nobs = sub_df.groupby([x_label]).size().reset_index()[0]
    ci_df = 1.96*std_df.div(np.sqrt(nobs), axis=0)
    return(mean_df, ci_df)

def plot_curve(sub_df, x_label, y_label, method, ax, only_ec=True):
    sub_df = sub_df[[x_label, y_label]]
    std_df = sub_df.groupby([x_label]).std().reset_index()
    mean_df = sub_df.groupby([x_label]).mean().reset_index()
    nobs = sub_df.groupby([x_label]).size().reset_index()[0]
    ci_df = 1.96*std_df.div(np.sqrt(nobs), axis=0)

    if method.endswith("-EC") and only_ec:
        method_ = method[:-3]
    else:
        method_ = method
    color = full2color[method_]

    if x_label=='False positive rate':
        ax.plot( np.concatenate([[0], mean_df[x_label],[1]]), np.concatenate([[0], mean_df[y_label],[1]]), 
             color=color,
             alpha=0.5, 
             label=method_, 
             linewidth=lw, 
             linestyle=linestyle[method_])   
    else:
        ax.plot(mean_df[x_label], mean_df[y_label], color=color, alpha=0.5, linewidth=lw, linestyle=linestyle[method_], label=method_)        
    ax.fill_between(x=mean_df[x_label], 
                    y1=(mean_df[y_label]-ci_df[y_label]), 
                    y2=(mean_df[y_label]+ci_df[y_label]), 
                    color=color, 
                    alpha=0.25)


def compute_rocs(raw_df):
    num = 1000
    grid = np.concatenate([ np.linspace(0,0.5/num, num=num) , np.linspace(1./num, (1-(1/num)), num=num), np.linspace((1-(0.5/num)),1, num=num) ])
    
    if 'N' in raw_df.columns:
        df_groups = raw_df.groupby(['N', 'tau', 'seed','method']).indices
    else:
        df_groups = raw_df.groupby(['seed','method']).indices
    roc_dfs = []
    for key, idx in tqdm(df_groups.items()):
        if len(key)==4:
            N, tau, seed, method = key
        else:
            seed, method = key
            tau = 0
            N = 0
        sub_df = raw_df.loc[idx]
        fpr, tpr, _ = roc_curve(y_true=sub_df['H'].values, y_score=(1-sub_df['q_value'].values))
        s = np.where( fpr[1:]>fpr[:-1])[0]
        condensed_fpr = np.concatenate([[0],fpr[s], [1]])
        condensed_tpr = np.concatenate([[0],tpr[s], [1]])
        vals = np.interp(x=grid, xp=condensed_fpr, fp=condensed_tpr)
        
        roc_dfs.append(pd.DataFrame({'N':N, 
                                      'tau':tau, 
                                      'seed':seed,
                                      'method':method,
                                      'False positive rate':np.concatenate([[0],grid, [1]]), 
                                      'True positive rate':np.concatenate([[0],vals, [1]])}))
    roc_df = pd.concat(roc_dfs, ignore_index=True)
    return(roc_df)


def plot_fdr(df, dirname, methods, only_ec=True):
    os.makedirs(dirname, exist_ok=True)
    Ns = np.unique(df['N'])
    num_N = len(Ns)
    
    taus = np.unique(df['tau'])
    num_tau = len(taus)
    plt.clf()
    fig, axs = plt.subplots(num_N, num_tau, figsize=(5*num_tau, 5*num_N))
    for i, N in enumerate(Ns):
        for j, tau in enumerate(taus):
            ax = axs[i,j]
            for m, method in enumerate(methods):
                sub_df = df[(df['N']==N)&(df['tau']==tau)&(df['method']==method)]
                plot_curve(sub_df, x_label="Nominal FDR", y_label="Observed FDR", method=method, ax=ax)
            ax.plot([0,1],[0,1], linestyle='--', color="black")
            ax.legend()
            ax.set_xlabel("Nominal FDR")
            ax.set_ylabel("Observed FDR")
            ax.set_title("# obs: {}, treatment strength: {}".format(N, tau))
    plt.tight_layout()
    plt.savefig(os.path.join(dirname,"fdr.pdf"), bbox_inches='tight')

def plot_roc(roc_df, dirname, methods):
    os.makedirs(dirname, exist_ok=True)
    Ns = np.unique(roc_df['N'])
    num_N = len(Ns)
    
    taus = np.unique(roc_df['tau'])
    num_tau = len(taus)
    plt.clf()
    fig, axs = plt.subplots(num_N, num_tau, figsize=(5*num_tau, 5*num_N))

    for i, N in enumerate(Ns):
        for j, tau in enumerate(taus):
            ax = axs[i,j]
            for m, method in enumerate(methods):
                sub_df = roc_df[(roc_df['N']==N)&(roc_df['tau']==tau)&(roc_df['method']==method)]
                plot_curve(sub_df, x_label="False positive rate", y_label="True positive rate", method=method, ax=ax)
            ax.plot([0,1],[0,1], linestyle='--', color="black")
            ax.legend()
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("# obs: {}, treatment strength: {}".format(N, tau))
    plt.tight_layout()
    plt.savefig(os.path.join(dirname,"roc.pdf"), bbox_inches='tight')


def load_dfs(folder):
    folder_names = [x for x in os.listdir(folder) if x.startswith('N_')]
    raw_dfs = []
    for folder_name in folder_names:
        _,N,_,tau,_,seed = folder_name.split('_')
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
    roc_df = compute_rocs(raw_df)

    dfs = []
    for folder_name in folder_names:
        _,N,_,tau,_,seed = folder_name.split('_')
        for model in all_methods:
            fname = os.path.join(folder, folder_name, model+".csv")
            if os.path.isfile(fname):
                df = pd.read_csv(fname, index_col=0)
                df.rename(columns={x:' '.join(x.split('.')) for x in df.columns}, inplace=True)
                df.rename(columns={"False discovery rate":"Observed FDR"}, inplace=True)
                df['seed'] = int(seed)
                df['N'] = int(N)
                df['tau'] = float(tau)
                df['method'] = model
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.fillna(0, inplace=True)
    df = df[["Nominal FDR", "Observed FDR", "N", "tau", "seed", "method"]]
    df.replace(abbrev2full, inplace=True)

    return(df, roc_df)

# colors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
colors = colormaps['tab10'](np.linspace(0, 1, num=8))

abbrev2full = {"causal_forest":"Causal forest", 
               "bart":"BART", 
               "causal2groups":"KC2G", 
               "causal2groups_ec":"KC2G-EC", 
               "additive_causal2groups":"AC2G", 
               "additive_causal2groups_ec":"AC2G-EC", 
               "nonadditive_causal2groups":"KC2G", 
               "nonadditive_causal2groups_ec":"KC2G-EC", 
               "FDRreg":"FDRreg"}

linestyle = {"Causal forest":"solid", 
             "BART":"dotted", 
             "KC2G":'dotted', 
             "KC2G-EC":'solid', 
             "AC2G":'solid', 
             "AC2G-EC":'dotted', 
             "FDRreg":'dashdot'}

full2color = {"KC2G":"tab:orange", 
              "AC2G":"tab:blue", 
              "Causal forest":"tab:brown", 
              "BART":"tab:pink", 
              "FDRreg":"tab:green", 
              "KC2G-EC":"tab:purple", 
              "AC2G-EC":"tab:red"}

if __name__=="main":
    # full2color = dict(zip(np.unique(list(abbrev2full.values())),colors))


    ## Non-linear simulations
    # folder = "./results/nonlinear/"
    # df_nonlinear, roc_df_nonlinear = load_dfs(folder)
    # dirname = "./plots/nonadditive"
    # plot_fdr(df_nonlinear, dirname, methods=["Causal forest", "BART", "FDRreg", 'AC2G-EC', 'KC2G-EC'])
    # plot_roc(roc_df_nonlinear, dirname, methods=["Causal forest", "BART", "FDRreg", 'AC2G', 'KC2G'])

    # ## Linear simulations
    # folder = "./results/linear/"
    # dirname = "./plots/additive"
    # df_linear, roc_df_linear = load_dfs(folder)
    # plot_fdr(df_linear, dirname, methods=["Causal forest", "BART",  "FDRreg", 'AC2G-EC', 'KC2G-EC'])
    # plot_roc(roc_df_linear, dirname, methods=["Causal forest", "BART", "FDRreg", 'AC2G', 'KC2G'])

    ## Nutlin simulations
    only_ec = True
    folder = "./results/nutlin/"
    dirname = "./plots/nutlin"
    os.makedirs(dirname, exist_ok=True)

    folder_names = [x for x in os.listdir(folder) if (x.startswith('pca_'))]
    dfs = []
    for folder_name in folder_names:
        *_,seed = folder_name.split('_')
        for model in all_methods:
            fname = os.path.join(folder, folder_name, model+".csv")
            if os.path.isfile(fname):
                df = pd.read_csv(fname, index_col=0)
                df.rename(columns={x:' '.join(x.split('.')) for x in df.columns}, inplace=True)
                df.rename(columns={"False discovery rate":"Observed FDR"}, inplace=True)
                df['seed'] = int(seed)
                df['method'] = model
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.fillna(0, inplace=True)
    df = df[["Nominal FDR", "Observed FDR","seed", "method"]]
    df.replace(abbrev2full, inplace=True)

    plt.close()
    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    methods=["Causal forest", "BART", "FDRreg", 'AC2G-EC', 'KC2G-EC']
    for m, method in enumerate(methods):
        sub_df = df[(df['method']==method)]
        sub_df = sub_df[["Nominal FDR", "Observed FDR"]]
        plot_curve(sub_df, x_label="Nominal FDR", y_label="Observed FDR", method=method, ax=axs)
    plt.plot([0,1],[0,1], linestyle='--', color="black")
    plt.legend()
    plt.xlabel("Nominal FDR")
    plt.ylabel("Observed FDR")
    # plt.title("Nutlin - PCA")
    plt.savefig(os.path.join(dirname,"fdr.pdf"), bbox_inches='tight')


    ## Nutlin ROC
    raw_dfs = []
    for folder_name in folder_names:
        *_, seed = folder_name.split('_')
        for model in all_methods:
            fname = os.path.join(folder, folder_name, model+"_raw.csv")
            if os.path.isfile(fname):
                raw_df = pd.read_csv(fname, index_col=0)
                raw_df.rename(columns={x:' '.join(x.split('.')) for x in raw_df.columns}, inplace=True)
                raw_df['seed'] = int(seed)
                raw_df['method'] = model
                raw_dfs.append(raw_df)

    raw_df = pd.concat(raw_dfs, ignore_index=True)
    raw_df.fillna(0, inplace=True)
    raw_df.replace(abbrev2full, inplace=True)
    roc_df = compute_rocs(raw_df)

    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    methods=["Causal forest", "BART", "FDRreg", 'AC2G', 'KC2G']
    for m, method in enumerate(methods):
        sub_df = roc_df[(roc_df['method']==method)]
        plot_curve(sub_df, x_label="False positive rate", y_label="True positive rate", method=method, ax=axs)
                    
    plt.plot([0,1],[0,1], linestyle='--', color="black")
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    # plt.title("Nutlin - PCA")
    plt.tight_layout()
    plt.savefig(os.path.join(dirname,"roc.pdf"), bbox_inches='tight')

    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2, 1, figsize=(5, 2*3.5))
    ax = axs[0]
    methods=["Causal forest", "BART", "FDRreg", 'AC2G-EC', 'KC2G-EC']
    for m, method in enumerate(methods):
        sub_df = df[(df['method']==method)]
        plot_curve(sub_df, x_label="Nominal FDR", y_label="Observed FDR", method=method, ax=ax)
    ax.plot([0,1],[0,1], linestyle='--', color="black")
    ax.set_xlabel("Nominal FDR")
    ax.set_ylabel("Observed FDR")
    ax.legend()

    ax = axs[1]
    methods=["Causal forest", "BART", "FDRreg", 'AC2G', 'KC2G']
    for m, method in enumerate(methods):
        sub_df = roc_df[(roc_df['method']==method)]
        plot_curve(sub_df, x_label="False positive rate", y_label="True positive rate", method=method, ax=ax)
    ax.plot([0,1],[0,1], linestyle='--', color="black")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    plt.savefig(os.path.join(dirname,"combined.pdf"), bbox_inches='tight')




    ## Plot combined
    plt.close()

    def combined_plot(df, roc_df, dirname):
        N = 1000
        taus = [1, 3]
        plt.clf()
        fig, axs = plt.subplots(2, 2, figsize=(5*2, 3.5*2))
        
        ## Plot FDRs
        for j, tau in enumerate(taus):
            ax = axs[0,j]
            for m, method in enumerate(methods):
                sub_df = df[(df['N']==N)&(df['tau']==tau)&(df['method']==method)]
                plot_curve(sub_df, x_label="Nominal FDR", y_label="Observed FDR", method=method, ax=ax)
            ax.plot([0,1],[0,1], linestyle='--', color="black")
            ax.set_xlabel("Nominal FDR")
            ax.set_ylabel("Observed FDR")
            ax.set_title("Treatment strength: {}".format(tau))
        axs[0,0].legend()

        ## Plot ROCs
        for j, tau in enumerate(taus):
            ax = axs[1,j]
            for m, method in enumerate(methods):
                sub_df = roc_df[(roc_df['N']==N)&(roc_df['tau']==tau)&(roc_df['method']==method)]
                plot_curve(sub_df, x_label="False positive rate", y_label="True positive rate", method=method, ax=ax)
            ax.plot([0,1],[0,1], linestyle='--', color="black")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
        plt.tight_layout()
        plt.savefig(os.path.join(dirname, "combined.pdf"), bbox_inches='tight')


    folder = "./results/nonlinear/"
    df_nonlinear, roc_df_nonlinear = load_dfs(folder)
    dirname = "./plots/nonadditive"
    combined_plot(df_nonlinear, roc_df_nonlinear, dirname)


    folder = "./results/linear/"
    dirname = "./plots/additive"
    df_linear, roc_df_linear = load_dfs(folder)
    combined_plot(df_linear, roc_df_linear, dirname)
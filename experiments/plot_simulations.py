import numpy as np
import pandas as pd
from causal2groups.postprocess import ResultsInterpreter
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

lw = 4
linestyle = {"Causal forest":"solid", 
             "BART":"dotted", 
             "NP-C2G":'dotted', 
             "NP-C2G-EC":'solid', 
             "Add-C2G":'solid', 
             "Add-C2G-EC":'dotted', 
             "FDRreg":'dashdot',
             "Frequentist":"solid", 
             "NP-Oracle":"solid"}

full2color = {"NP-C2G":"tab:orange", 
              "Add-C2G":"tab:blue", 
              "Causal forest":"tab:brown", 
              "BART":"tab:pink", 
              "FDRreg":"tab:green", 
              "NP-C2G-EC":"tab:orange", 
              "Add-C2G-EC":"tab:blue", 
              "Frequentist":"tab:red", 
              "NP-Oracle":"black"}


def plot_curve(x, y_mean, y_ci, method, ax):
    color = full2color[method]
    skip = 5
    
    ax.plot(x[::skip],y_mean[::skip], color=color, alpha=0.5, linewidth=lw, linestyle='-', label=method) 
    ax.fill_between(x=x[::skip], 
                    y1=(y_mean[::skip]-y_ci[::skip]), 
                    y2=(y_mean[::skip]+y_ci[::skip]), 
                    color=color, 
                    alpha=0.25)
    
res = ResultsInterpreter("./results")


Ns = [1000, 10000]
taus = [1, 3, 5]
settings = ['additive', 'nonadditive']
metric2xlabel = {"fdr":"Nominal FDR", "power":"Nominal FDR", "roc":"False positive rate"}
metric2ylabel = {"fdr":"Observed FDR", "power":"Valid power", "roc":"True positive rate"}
methods=["Causal forest", "BART", "FDRreg", "Frequentist", 'Add-C2G', 'NP-C2G', 'NP-Oracle']

for setting in settings:
    for metric in ["fdr", "power", "roc"]:
        os.makedirs(os.path.join("./plots",setting), exist_ok=True) ## Make the directory
        df = res.get_df(metric)
        xlabel = metric2xlabel[metric]
        ylabel = metric2ylabel[metric]
        plt.clf()
        fig, axs = plt.subplots(2, 3, figsize=(5*3, 3.5*2))
        for i, N in enumerate(Ns):
            for j, tau in enumerate(taus):
                ax = axs[i,j]
                for m, method in enumerate(methods):
                    sub_df = df[(df['N']==N)&(df['tau']==tau)&(df['method']==method)&(df['Setting']==setting)]
                    plot_curve(sub_df[xlabel], sub_df[ylabel+"_mean"], sub_df[ylabel+"_CI"], method, ax)                
                ax.plot([0,1],[0,1], linestyle='--', color="black")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title("Treatment strength: {}".format(tau))
            ax = axs[i,0]
            ax.legend(loc='lower right')
            ax.annotate("N={}".format(N), xy=(0, 0.2), xytext=(-ax.yaxis.labelpad - 15, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points', ha='center',
                        size='large', rotation=90)
        plt.tight_layout()
        fig.subplots_adjust(left=0.1, top=1)
        plt.savefig("./plots/{}/{}.pdf".format(setting, metric), bbox_inches='tight')
        

## Nutlin
setting = 'nutlin'
os.makedirs(os.path.join("./plots",setting), exist_ok=True) ## Make the directory
plt.clf()
fig, axs = plt.subplots(1, 2, figsize=(2*5, 3.5))
for i, metric in enumerate(["fdr", "power"]):
    df = res.get_df(metric)
    xlabel = metric2xlabel[metric]
    ylabel = metric2ylabel[metric]
    ax = axs[i]
    for m, method in enumerate(methods[:-1]):
        method = method.replace("-EC","") if metric=='roc' else method
        sub_df = df[(df['method']==method)&(df['Setting']==setting)]
        plot_curve(sub_df[xlabel], sub_df[ylabel+"_mean"], sub_df[ylabel+"_CI"], method, ax)                
    ax.plot([0,1],[0,1], linestyle='--', color="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
axs[0].legend(loc='upper left')
plt.tight_layout()
plt.savefig("./plots/{}/combined.pdf".format(setting), bbox_inches='tight')


fin_lookup = {}
fin_lookup.update(res.fdr_lookup(0.1))
fin_lookup.update(res.power_lookup(0.1))
fin_lookup.update(res.auc_lookup())

for N in [1000]:
    for metric in ["Observed FDR", 'Valid power']:
        print("N = {}, metric = {}".format(N, metric))
        print(r"\begin{tabular}{||c|c|c|c|c|c|c||}")
        print(r"\hline")
        print(r"& \multicolumn{3}{|c|}{Additive} & \multicolumn{3}{|c|}{Nonadditive} \\")
        print(r"\hline")
        print(r"Method & $\tau=1$ & $\tau=3$ & $\tau=5$ & $\tau=1$ & $\tau=3$ & $\tau=5$ \\")
        print(r"\hline")
        for method in ["Frequentist", 'Add-C2G', 'NP-C2G', "NP-Oracle"]:
            row = [method]
            for setting in ["additive", "nonadditive"]:
                for tau in [1.,3.,5.]:
                    mean = fin_lookup[metric+"_mean"][N, tau, method, setting]
                    ci = fin_lookup[metric+"_CI"][N, tau, method, setting]
                    entry = "{}±{}".format(mean, ci)
                    if fin_lookup[metric+"_bold"][N, tau, method, setting]:
                        entry = r"\textbf{ "+ entry + " }"

                    row.append(entry)
            print(" & ".join(row) + r" \\")
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\label{table:fdr-at-level-simulations}")
        print(r"\end{table}")
        print()



### ITE
ite_dfs = []
base_folder = "./results"
for setting in ["additive", "nonadditive"]:
    folder = os.path.join(base_folder, setting)
    folder_names = [x for x in os.listdir(folder) if x.startswith('N_')]
    for folder_name in tqdm(folder_names):
        _,N,tau,_,seed = folder_name.split('_')
        tau = float(tau[3:])
        seed = int(seed)
        N = int(N)
        for model in ["additive_causal2groups", "nonadditive_causal2groups"]:
            fname = os.path.join(folder, folder_name, model+"_ite.csv")
            if os.path.isfile(fname):
                ite_df = pd.read_csv(fname, index_col=0)                  
                ite_df['method'] = model
                ite_df.reset_index(names="data_index", inplace=True)
                ite_df['seed'] = seed
                ite_df['N'] = N
                ite_df['tau'] = tau
                ite_df['setting'] = setting
                ite_dfs.append(ite_df)

ite_df = pd.concat(ite_dfs, ignore_index=True)
ref_df = pd.read_csv("./results/oracle.csv")
interval_df = ite_df[ite_df['method']=='nonadditive_causal2groups'].merge(ref_df, on=["setting", "N", "tau", "seed", "data_index"], suffixes=('_pred', '_oracle'))
interval_df = interval_df[['N', 'tau', 'seed', 'setting', 'data_index', 'ITE lower bound_pred', 'ITE upper bound_pred', 'ITE lower bound_oracle', 'ITE upper bound_oracle']]
ate_df = interval_df.groupby(['N', 'tau', 'seed', 'setting']).mean().reset_index()

upper = np.minimum(ate_df['ITE upper bound_pred'],ate_df['ITE upper bound_oracle'])
lower = np.maximum(ate_df['ITE lower bound_pred'],ate_df['ITE lower bound_oracle'])
intersection_width = np.clip(upper - lower, a_min=0.0, a_max=None)
pred_width = ate_df['ITE upper bound_pred']-ate_df['ITE lower bound_pred']
oracle_width = ate_df['ITE upper bound_oracle']-ate_df['ITE lower bound_oracle']

ate_df['Overlap coefficient'] = intersection_width/(np.minimum(pred_width, oracle_width))
ate_df['Jaccard index'] = intersection_width/(pred_width + oracle_width - intersection_width)

mean_df = ate_df.groupby(['setting', 'N', 'tau']).mean().reset_index()
std_df = ate_df.groupby(['setting', 'N', 'tau']).std().reset_index()
size_df = ate_df.groupby(['setting', 'N', 'tau']).size().reset_index()
res_df = mean_df.merge(std_df, on=["N", "tau", "setting"], suffixes=('_mean', '_stdv'))
res_df = res_df.merge(size_df, on=["N", "tau", "setting"])
res_df.rename(columns={0:"count"}, inplace=True)
out_df = res_df[['setting', 'N', 'tau', 'Jaccard index_mean', 'Jaccard index_stdv', 'count']].copy()
out_df['Jaccard index_CI'] = 1.96*out_df['Jaccard index_stdv']/np.sqrt(out_df['count'])
out_df['method'] = "NP-C2G"

out_df['Jaccard index_mean'] = out_df['Jaccard index_mean'].round(2)
out_df['Jaccard index_CI'] = out_df['Jaccard index_CI'].round(2)
lookup = out_df.set_index(['N', 'tau', 'method', 'setting']).to_dict()


metric = "Jaccard index"
method = 'NP-C2G'
print(r"\begin{tabular}{||c|c|c|c|c|c|c||}")
print(r"\hline")
print(r"& \multicolumn{3}{|c|}{Additive} & \multicolumn{3}{|c|}{Nonadditive} \\")
print(r"\hline")
print(r"N & $\tau=1$ & $\tau=3$ & $\tau=5$ & $\tau=1$ & $\tau=3$ & $\tau=5$ \\")
print(r"\hline")
for N in [1000, 10000]:
    row = [str(N)]
    for setting in ["additive", "nonadditive"]:
        for tau in [1.,3.,5.]:
            mean = lookup[metric+"_mean"][N, tau, method, setting]
            ci = lookup[metric+"_CI"][N, tau, method, setting]
            entry = "{}±{}".format(mean, ci)
            row.append(entry)
    print(" & ".join(row) + r" \\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\label{table:nonadditive-jaccard-ite}")
print(r"\end{table}")
print()

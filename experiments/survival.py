import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from matplotlib import pyplot as plt
from causal2groups.npc2g import KernelNonparametricCausal2G
import distinctipy
from scipy.stats import fisher_exact, false_discovery_control
from tqdm import tqdm
import os

os.makedirs("./plots/survival", exist_ok=True)

np.random.seed(200)

mut_df = pd.read_csv("./data/tmb_mskcc_2018/mutations.csv")
patient_mut_embeds = pd.read_csv("./data/tmb_mskcc_2018/patient_mut_embeds.csv", index_col=0)
patient_mut_embeds = patient_mut_embeds.rename(columns={k:"mut_{}".format(k) for k in patient_mut_embeds.columns})
patient_mut_embeds.reset_index(names="Patient ID", inplace=True)
gene_df = pd.read_csv("./data/tmb_mskcc_2018/genes.csv")
gene_id2gene_name = dict(zip(gene_df['gene_id'], gene_df['gene_name']))

clinical_df = pd.read_csv("./data/tmb_mskcc_2018/tmb_mskcc_2018_clinical_data.tsv", sep="\t")

keeper_covariates = ['Age Group at Diagnosis in Years', 'Cancer Type', 'Sample Type', 'Sex']
treatment_variable = 'Drug Type'
outcome_variable = 'Overall Survival (Months)'

one_hots = []
columns = []
n = clinical_df.shape[0]
for v in keeper_covariates:
    vals, inverse = np.unique(clinical_df[v], return_inverse=True)
    columns.append(v + '_' + vals)
    X = np.zeros((n, vals.shape[0]), dtype=int)
    X[np.arange(n), inverse] = 1
    one_hots.append(X)

clinical_feature_df = pd.DataFrame(data=np.concatenate(one_hots, axis=1), columns=np.concatenate(columns), index=clinical_df['Patient ID']).reset_index()
clinical_feature_df = clinical_feature_df.merge(patient_mut_embeds, on="Patient ID")
clinical_feature_df['Treatment'] = clinical_df[treatment_variable]
clinical_feature_df['Overall Survival (Months)'] = clinical_df[outcome_variable]
clinical_feature_df['Overall Survival Status'] = np.where(clinical_df['Overall Survival Status']=="1:DECEASED", 1, 0)

drug_types = ["CTLA4", "Combo"]
kms = []
for drug_type in drug_types:
    mask = clinical_df['Drug Type'].values==drug_type
    T = clinical_feature_df['Overall Survival (Months)'].values[mask]
    E = clinical_feature_df['Overall Survival Status'].values[mask]
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E)
    kms.append(kmf)

def columns_to_keep(df):
    events = null_df['Overall Survival Status'].astype(bool)
    stdvs_1 = null_df[events].std()
    stdvs_0 = null_df[~events].std()
    keeper_cols = np.union1d(np.intersect1d(stdvs_1.index[stdvs_1.values>0], stdvs_0.index[stdvs_0.values>0]), ['Overall Survival Status'])
    return(keeper_cols)

cph = CoxPHFitter(penalizer=0.05)
null_df = clinical_feature_df[clinical_feature_df['Treatment']=='CTLA4'].drop(columns=['Patient ID', 'Treatment'])
keeper_cols = columns_to_keep(null_df)
null_df = null_df[keeper_cols]

cph.fit(null_df, duration_col='Overall Survival (Months)', event_col='Overall Survival Status')

treat_df = clinical_feature_df[clinical_feature_df['Treatment']=='Combo'].drop(columns=['Patient ID', 'Treatment'])
treat_df = treat_df[keeper_cols]

treat_preds = cph.predict_expectation(treat_df).values
null_preds = cph.predict_expectation(null_df).values


# Calculate the z-scores for every sample using the tansey residual
def z_score(model, X, T, E, max_survival_time):
    from scipy.stats import norm
    grid = np.arange(1, max_survival_time+1)
    surv = model.predict_survival_function(X, times=grid).T
    probs = surv.values[:,:-1] - surv.values[:,1:]
    probs = np.hstack([np.zeros((surv.shape[0], 1)), probs])
    T = np.copy(T.astype(int))
    for i in range(X.shape[0]):
        if E[i] == 0:
            conditional_prob = probs[i, T[i]+1:].sum()
            if conditional_prob > 0:
                T[i] = int(np.round((grid[T[i]+1:] * probs[i, T[i]+1:]).sum() / probs[i, T[i]+1:].sum()))
    cdf_vals = 1-surv.values[np.arange(X.shape[0]), T]
    return norm.ppf(cdf_vals.clip(1e-10, 1-1e-10))


null_z_scores = z_score(cph, X=null_df, T=null_df['Overall Survival (Months)'].values, E=null_df['Overall Survival Status'].values, max_survival_time=120)
treat_z_scores = z_score(cph, X=treat_df, T=treat_df['Overall Survival (Months)'].values, E=treat_df['Overall Survival Status'].values, max_survival_time=120)

fig, axs = plt.subplots(1,2, figsize=(10,4))

for drug_type, kmf in zip(["Null group", "Treatment group"], kms):
    axs[0].plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate.values, label=drug_type)
    axs[0].fill_between(x=kmf.confidence_interval_survival_function_.index.values, 
                     y1=kmf.confidence_interval_survival_function_['KM_estimate_lower_0.95'].values,
                     y2=kmf.confidence_interval_survival_function_['KM_estimate_upper_0.95'].values, alpha=0.25)
axs[0].legend()
axs[0].set_xlabel('Overall Survival (Months)')
axs[0].set_ylabel('Kaplan–Meier estimate')



_, bins, _ = axs[1].hist(treat_z_scores, bins=50, density=True, alpha=0.5, color="tab:orange", label='Treatment group')
axs[1].hist(null_z_scores, bins=bins, density=True, alpha=0.5, color="tab:blue", label='Null group', zorder=0)
axs[1].legend()
axs[1].set_xlabel("Outcome")
axs[1].set_ylabel("Density")
plt.tight_layout()
plt.savefig("./plots/survival/km_zscore.pdf", bbox_inches='tight')


full_index = np.concatenate([null_df.index, treat_df.index])
c2g = KernelNonparametricCausal2G(kernel_n_neighbors=[50, 100, 200], kernel_bandwidth_neighbor_fracs=np.logspace(-3,0, num=10), verbose=True)
X = pd.concat([null_df, treat_df], ignore_index=True).drop(columns=["Overall Survival (Months)", 'Overall Survival Status']).values
T = np.concatenate([ [0]*null_df.shape[0], [1]*treat_df.shape[0]  ])
Y = np.concatenate([null_z_scores, treat_z_scores])
c2g.fit(X=X, Y=Y, T=T)



upper, lower = c2g.predict_ite()
upper, lower = np.maximum(upper, lower), np.minimum(upper, lower)

upper_treat, lower_treat = upper[T==1], lower[T==1]


## keeper genes
def calculate_q_values(sig_patient_idx, alternative):    
    treat_mut_df = mut_df[mut_df['patient_idx'].isin(treat_df.index)].reset_index(drop=True)
    treat_mut_df['Selected'] = treat_mut_df['patient_idx'].isin(sig_patient_idx)
    n_patients = treat_mut_df['patient_idx'].nunique()
    temp_df = treat_mut_df.groupby('gene_id').nunique().reset_index()
    keeper_genes = temp_df[temp_df['patient_idx']==n_patients]['gene_id'].values
    treat_mut_df = treat_mut_df[treat_mut_df['gene_id'].isin(keeper_genes)].reset_index(drop=True)
    groupings = treat_mut_df.groupby('gene_id').groups
    contingency_tables = {}
    for gene_id, idx in groupings.items():
        sub_df = treat_mut_df.iloc[idx]
        contingency_tables[gene_id2gene_name[gene_id]] = pd.crosstab(sub_df['observed'], sub_df['Selected'].astype(int))
    
    test_results = []
    for gene_name, table in contingency_tables.items():
        if table.shape == (2,2):
            r = fisher_exact(table.loc[[1,0],[1,0]].values, alternative=alternative)
            test_results.append({"gene":gene_name, "statistic":r.statistic, "p_value":r.pvalue})
    if len(test_results) > 0:
        test_df = pd.DataFrame(test_results)
        test_df['q_value'] = false_discovery_control(test_df['p_value'])
        return(test_df)
    else:
        return(None)

alphas = np.linspace(0, 0.25, num=200)
selections = c2g.select(fdr_levels=alphas, empirical_control=True)
res_dfs = []
for i, alpha in tqdm(enumerate(alphas), total=len(alphas)):
    rdf = calculate_q_values(full_index[selections[i]], alternative='greater')
    if rdf is not None:
        rdf['alpha'] = alpha
        res_dfs.append(rdf)
greater_df = pd.concat(res_dfs, ignore_index=True)

res_dfs = []
for i,alpha in tqdm(enumerate(alphas), total=len(alphas)):
    rdf = calculate_q_values(full_index[selections[i]], alternative='less')
    if rdf is not None:
        rdf['alpha'] = alpha
        res_dfs.append(rdf)
less_df = pd.concat(res_dfs, ignore_index=True)


thresh = -np.log10(0.1)
fig, axs = plt.subplots(1,2, figsize=(12,5))
alternative = ['More', 'Less']
jj = 0
colors = distinctipy.get_colors(12, colorblind_type = "Deuteranomaly")
for i, res_df in enumerate([greater_df, less_df]):
    res_df['neg log_10 q_value'] = -np.log10(res_df['q_value'])
    pivot_df = res_df.pivot(index="gene", columns="alpha", values="neg log_10 q_value")
    sub_df = pivot_df[(pivot_df>thresh).any(axis=1)]
    for gene_name, v in zip(sub_df.index.values, sub_df.values):
        axs[i].plot(sub_df.columns.values, v, color=colors[jj], label=gene_name)
        jj+=1
    axs[i].set_xlabel("Nominal FDR level")
    axs[i].set_ylabel(r"- $\log_{10}$(q value)")
    axs[i].legend()
    axs[i].set_title('{} favorable'.format(alternative[i]))
    
plt.tight_layout()
plt.savefig("./plots/survival/q_val.pdf", bbox_inches='tight')


## Greater than
pivot_df = greater_df.pivot(index="gene", columns="alpha", values="neg log_10 q_value")
sub_df = pivot_df[(pivot_df>thresh).any(axis=1)]
g_largest_q_value = sub_df.max(axis=1)
g_first_q_lessthan_val = (sub_df > thresh).idxmax(axis=1)

## Less than
pivot_df = less_df.pivot(index="gene", columns="alpha", values="neg log_10 q_value")
sub_df = pivot_df[(pivot_df>thresh).any(axis=1)]
l_largest_q_value = sub_df.max(axis=1)
l_first_q_lessthan_val = (sub_df > thresh).idxmax(axis=1)

print(' & '.join(["Mutation"] + list(np.concatenate([g_largest_q_value.index, l_largest_q_value.index]))) + r" \\")
print(r"\hline")
print(' & '.join([r"Smallest $\alpha$"] + list(np.round(np.concatenate([g_first_q_lessthan_val.values, l_first_q_lessthan_val.values]),3).astype(str))) + r" \\")
print(r"\hline")
print(' & '.join([r"Largest -$\log_{10}(q)$"] + list(np.round(np.concatenate([g_largest_q_value.values, l_largest_q_value.values]),3).astype(str))) + r" \\")


mut_df = pd.read_csv("./data/tmb_mskcc_2018/mutations.csv")
all_patient_idx = np.concatenate([null_df.index, treat_df.index])
mut_df = mut_df[mut_df['patient_idx'].isin(all_patient_idx)].reset_index(drop=True)
mut_df['gene_name'] =  mut_df['gene_id'].map(gene_id2gene_name)
interesting_genes = np.concatenate([g_largest_q_value.index, l_largest_q_value.index])
mut_df = mut_df[mut_df['gene_name'].isin(interesting_genes)]
mut_df = mut_df[mut_df['observed']==1].reset_index(drop=True)
row = ['CARE interval']
for gene in interesting_genes:
    sub_df = mut_df[mut_df['gene_name']==gene]
    mask = np.isin(all_patient_idx,sub_df['patient_idx'])
    ub = np.round(np.mean(upper[mask]),1)
    lb = np.round(np.mean(lower[mask]),1)
    row += ['({}, {})'.format(lb, ub)]
print(' & '.join(row) + r'\\')

row = ['CERPF']
for gene in interesting_genes:
    sub_df = mut_df[mut_df['gene_name']==gene]
    mask = np.isin(all_patient_idx, sub_df['patient_idx'])&(T==1)
    val = np.round(np.mean(c2g.pi_star[mask]),2)
    row += ['{}'.format(val)]
print(' & '.join(row) + r'\\')

print("Prior ERPF: {0:.2f}, Posterior ERPF: {1:.2f}".format(np.mean(c2g.pi_star[T==1]), np.mean((1-c2g.null_posterior)[T==1])))
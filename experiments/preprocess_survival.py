import pandas as pd
import numpy as np
import mygene

### Get all the genes in order
clinical_df = pd.read_csv("./data/tmb_mskcc_2018/tmb_mskcc_2018_clinical_data.tsv", sep='\t')
x = clinical_df.nunique()
keep_cols = x.index[x.values>1]
clinical_df = clinical_df[keep_cols]
gene_df = pd.read_csv("./data/tmb_mskcc_2018/data_gene_panel_matrix.txt", sep='\t')
mut_df = pd.read_csv("./data/tmb_mskcc_2018/data_mutations.txt", sep='\t')
mut_df['Entrez_Gene_Id'] = mut_df['Entrez_Gene_Id'].astype(int)
mut_df = mut_df[['Tumor_Sample_Barcode', 'Hugo_Symbol', 'Entrez_Gene_Id', 'Variant_Classification', ]]
merge_df = clinical_df.merge(mut_df, left_on="Sample ID", right_on="Tumor_Sample_Barcode")

panels = np.unique(clinical_df['Gene Panel'])
impact2genes = {}
for panel in panels:
    pdf = pd.read_csv("./data/tmb_mskcc_2018/data_gene_panel_myb_{}.txt".format(panel.upper()), sep="\t", skiprows=2, skipfooter=1, engine='python')
    v = pdf.columns.values
    v[0] = 'ABL1'
    impact2genes[panel] = v

mg = mygene.MyGeneInfo()
genes = np.unique(np.concatenate([impact2genes['IMPACT341'], impact2genes['IMPACT410'], impact2genes['IMPACT468']]))

gene2entrez = dict(zip(mut_df['Hugo_Symbol'].values, mut_df['Entrez_Gene_Id'].values.astype(int)))
obs_gene_names = np.unique(mut_df['Hugo_Symbol'].values)
obs_gene_ids = np.unique(mut_df['Entrez_Gene_Id'].values)

rem_genes = np.setdiff1d(genes, obs_gene_names)
queries = [mg.query(gene, species='human') for gene in rem_genes]
hits = [[(x['symbol'], int(x['entrezgene'])) for x in q['hits'] if 'entrezgene' in x] for q in queries]

add_mask = [np.sum([(y in obs_gene_ids) for x,y in h])==1 for h in hits]
remove_mask = [np.sum([(y in obs_gene_ids) for x,y in h])>1 for h in hits]
remove_ids = np.concatenate([ [y for x,y in h]  for h,v in zip(hits, remove_mask) if v])
add_gene2entrez = dict([[(x,y) for x,y in h if (y in obs_gene_ids)][0] for h,v in zip(hits, add_mask) if v])
remove_genes = [k for k,v in gene2entrez.items() if v in remove_ids]
gene2entrez.update(add_gene2entrez)
for gene in remove_genes:
    del gene2entrez[gene]

impact2ids = {panel:np.array([ gene2entrez[gene] for gene in genes if gene in gene2entrez])  for panel, genes in impact2genes.items()}

all_gene_ids = np.unique(list(gene2entrez.values()))
groupings = merge_df.groupby(["Patient ID", 'Gene Panel']).groups

dfs = []
for (patient_id, panel), idx in groupings.items():
    gene_ids = merge_df.loc[idx, "Entrez_Gene_Id"].values
    panel_genes = impact2ids[panel]
    curr_row = np.empty(all_gene_ids.shape)
    curr_row[:] = np.nan
    missing_mask = np.isin(all_gene_ids, panel_genes)
    curr_row[missing_mask] = 0
    included_mask = np.isin(all_gene_ids, gene_ids)
    curr_row[included_mask] = 1
    dfs.append(pd.DataFrame(data=curr_row[np.newaxis, :], columns=all_gene_ids, index=[patient_id]))

## Missing patients
missing_df = clinical_df[~clinical_df['Patient ID'].isin(merge_df['Patient ID'])]
for _, row in missing_df.iterrows():
    panel_genes = impact2ids[row['Gene Panel']]
    curr_row = np.empty(all_gene_ids.shape)
    curr_row[:] = np.nan
    missing_mask = np.isin(all_gene_ids, panel_genes)
    curr_row[missing_mask] = 0
    dfs.append(pd.DataFrame(data=curr_row[np.newaxis, :], columns=all_gene_ids, index=[row['Patient ID']]))

df = pd.concat(dfs)
df.sort_index(inplace=True)

df['patient_id'] = df.index
df_melt = df.melt(id_vars="patient_id", value_name="observed", var_name="gene_id")
_, df_melt['patient_idx'] = np.unique(df_melt['patient_id'], return_inverse=True)
_, df_melt['gene_idx'] = np.unique(df_melt['gene_id'], return_inverse=True)
df_melt = df_melt[~df_melt['observed'].isna()].reset_index(drop=True)
df_melt['observed'] = df_melt['observed'].astype(int)
df_melt.to_csv("./data/tmb_mskcc_2018/mutations.csv", index=0)

gene_df = pd.DataFrame({'gene_name':list(gene2entrez.keys()), 'gene_id':list(gene2entrez.values())})
gene_df.to_csv("./data/tmb_mskcc_2018/genes.csv", index=0)


#### Run Bernoulli matrix factorization
from causal2groups.factor_model import BernoulliFactorModel

seed = 100
np.random.seed(seed)

df = pd.read_csv("./data/tmb_mskcc_2018/mutations.csv")
nunique = df.nunique()
n_rows = nunique['patient_idx']
n_cols = nunique['gene_idx']

row_idx = df['patient_idx'].values
col_idx = df['gene_idx'].values
obs = df['observed'].values

patient_idx2patient_id = dict(zip(df['patient_idx'], df['patient_id']))
gene_idx2gene_id = dict(zip(df['gene_idx'], df['gene_id']))

batch_size = 20000
n_epochs = 300
n_steps = n_epochs*int(df.shape[0]/batch_size)

bfm = BernoulliFactorModel(n_rows=n_rows, n_cols=n_cols, n_steps=n_steps, batch_size=batch_size, seed=seed)
all_lls, losses = bfm.fit_via_cv(row_ids=row_idx, col_ids=col_idx, obs=obs, dim_list=[2, 5, 10, 50], n_folds=3)
probs = bfm.predict(row_idx, col_idx, prob_transform=True)


patient_mut_embeds = pd.DataFrame(data=bfm.row_embed, index=[patient_idx2patient_id[x] for x in range(bfm.row_embed.shape[0])])
patient_mut_embeds.to_csv("./data/tmb_mskcc_2018/patient_mut_embeds.csv")

gene_mut_embeds = pd.DataFrame(data=bfm.col_embed, index=[gene_idx2gene_id[x] for x in range(bfm.col_embed.shape[0])])
gene_mut_embeds.to_csv("./data/tmb_mskcc_2018/gene_mut_embeds.csv")
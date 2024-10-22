import numpy as np
import pandas as pd
from scipy.special import expit as ilogit

def pca(X, pc=50):
    from sklearn.decomposition import PCA
    return PCA(n_components=pc).fit_transform(X)

class NonadditiveSimulatedData:
    def __init__(self, P:int, tau:float, seed:int, sigma:float=1, v:float=1):
        self.rng = np.random.default_rng(seed)
        self.P = P
        self.tau = tau
        self.sigma = sigma
        self.v = v

        self.beta = self.rng.normal(0, sigma/np.sqrt(P), size=P)
        self.gamma = self.rng.normal(0, sigma/np.sqrt(P), size=P)
        self.theta = self.rng.normal(0, sigma/np.sqrt(P), size=P)
        self.c = abs(self.rng.normal(0, 2)) # half gaussian to ensure positive correlation between T and Y

    def generate_data(self, N:int):
        X = self.rng.normal(0, 1/np.sqrt(self.P), size=(N,self.P))
        W = X.dot(self.gamma)    # treatment propensity
        T = self.rng.binomial(1, ilogit(W), size=N)
        H_prob = ilogit(X.dot(self.beta))    # effective/response propensity
        H = T & self.rng.binomial(1, H_prob, size=N)
        interactions = (self.rng.random(size=(1,X.shape[1], X.shape[1])) <= 0.1) * X[:,None] * X[:,:,None]
        interactions = (self.rng.standard_t(3, size=(1,X.shape[1],X.shape[1])) * interactions).sum(axis=-1).sum(axis=-1)
        Y = self.rng.normal(np.log1p(np.exp(self.c * W + X.dot(self.theta) + self.tau * H + interactions)), self.v, size=N)
        return(X, Y, T, H, H_prob)
    
    def generate_conditional_data(self, x:np.ndarray, n:int):
        X_ = np.tile(x,n).reshape(n,-1)
        W = X_.dot(self.gamma)    # treatment propensity
        T = self.rng.binomial(1, ilogit(W), size=n)
        H_prob = ilogit(X_.dot(self.beta))    # effective/response propensity
        H = T & self.rng.binomial(1, H_prob, size=n)
        interactions = (self.rng.random(size=(1,X_.shape[1], X_.shape[1])) <= 0.1) * X_[:,None] * X_[:,:,None]
        interactions = (self.rng.standard_t(3, size=(1,X_.shape[1],X_.shape[1])) * interactions).sum(axis=-1).sum(axis=-1)
        Y = self.rng.normal(np.log1p(np.exp(self.c * W + X_.dot(self.theta) + self.tau * H + interactions)), self.v, size=n)
        return(Y, T, H, H_prob)
    

class AdditiveSimulatedData:
    def __init__(self, P:int, tau:float, seed:int, sigma:float=1, v:float=1):
        self.rng = np.random.default_rng(seed)
        self.P = P
        self.tau = tau
        self.sigma = sigma
        self.v = v

        self.beta = self.rng.normal(0, sigma/np.sqrt(P), size=P)
        self.gamma = self.rng.normal(0, sigma/np.sqrt(P), size=P)
        self.theta = self.rng.normal(0, sigma/np.sqrt(P), size=P)

    def generate_data(self, N:int):
        T = self.rng.random(size=N) <= 0.5 # Randomized treatment assignment
        X = self.rng.normal(0, 1/np.sqrt(self.P), size=(N,self.P))

        mu_0 = X.dot(self.beta)
        mu_1 = mu_0 + self.tau*(np.abs(X).dot(np.abs(self.gamma))) # treatment effect
        H_prob = ilogit(X.dot(self.theta))

        H = T & (self.rng.random(size=H_prob.shape) <= H_prob)

        Y_null = self.rng.normal(mu_0, scale=self.v)
        Y_effect = self.rng.normal(mu_1, scale=self.v)

        Y = np.where(H==1, Y_effect, Y_null)
        return(X, Y, T, H, H_prob)
    
    def prior_prob(self, X:np.ndarray):
        H_prob = ilogit(X.dot(self.theta))
        return(H_prob)

    def null_mean(self, X:np.ndarray):
        mu_0 = X.dot(self.beta)
        return(mu_0)

    def alt_mean(self, X:np.ndarray):
        mu_0 = self.null_mean(X)
        mu_1 = mu_0 + self.tau*(np.abs(X).dot(np.abs(self.gamma)))
        return(mu_1)

class GDSCSemiSynthetic:
    def __init__(self, 
                 features_df:pd.DataFrame, 
                 outcomes_df:pd.DataFrame, 
                 drug_df:pd.DataFrame, 
                 drug:str, 
                 mutations:list[str], 
                 conditions:list[float], 
                 seed:int):
        
        self.rng = np.random.default_rng(seed)

        ## Lookup drug id
        drugname2id = dict(zip(drug_df['Drug Name'],drug_df['Drug ID'] ))
        drug_id = drugname2id[drug]

        exp_cols = [col for col in features_df.columns if 'EXP:' in col]   # expressions columns
        all_cell_lines = features_df.index.unique()    # all the cell lines with features available
        
        X = []
        Y = []
        T = []
        H = []

        drug_outcome:pd.DataFrame = outcomes_df[outcomes_df['DRUG_ID']==drug_id].reset_index(drop=True) # all tests of this drug
        cell_lines = drug_outcome['CELL_LINE_NAME'].unique() # all cell lines been tested for this drug

        # treatment assignment based on prior knowledge
        df = features_df[features_df.index.isin(cell_lines)]
        
        # select effective cell lines with targeted mutations based on prior knowledge
        treated_cell_lines = []
        for mut, cond in zip(mutations, conditions):
            treated_cell_lines += list(df[df[f"MUT:{mut}"] == cond].index.values)
        treated_cell_lines = set(treated_cell_lines)

        for index, row in drug_outcome.iterrows():
            cell_line = row['CELL_LINE_NAME']
            if cell_line in all_cell_lines: # if we have features available for this cell line
                if cell_line in treated_cell_lines: # effective
                    T.append(1)
                    H.append(1)
                else:   # non effective
                    T.append(0)
                    H.append(0)
                Y.append(np.array(row['z_dose0']))    # z-score of maximum doses
                X.append(np.array(df[df.index==cell_line][exp_cols].values)[0])     # use gene experessions of cellline as covariates

        self.T = np.array(T)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.H = np.array(H)
        self.X_pca = pca(X)

        index = exp_cols.index('EXP:GAPDH')
        W:np.ndarray = self.X[:, index]
        self.W = (W - W.mean()) / W.std()

        # 09/15: use top 100 var raw features + GAPDH(normalized, treatment propensity W)
        indices = np.argsort(np.var(self.X, axis=0))[-100:]
        new_X = np.column_stack((self.X[:, indices], self.W))
        self.new_X = (new_X - new_X.mean(axis=0)) / new_X.std(axis=0)
        self.col_names = [exp_cols[i] for i in np.append(indices, index)]


    def generate_data(self, pca:bool):
        gamma = 1
        T_bias = self.rng.binomial(1, ilogit(self.W * gamma), size=self.Y.shape[0])
        T_bias[self.H == 1] = self.T[self.H == 1]  # keep effective the same, only mix more noneffective into treated
        Y_tilde = self.Y - gamma * self.W
        
        X = self.X_pca if pca else self.X
        Y = Y_tilde
        T = T_bias
        H = self.H
        return(X, Y, T, H)
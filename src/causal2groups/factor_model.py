import numpy as np
from jax import jit, Array
import jax.numpy as jnp
from jax.scipy.special import expit
from jax.random import PRNGKey, split
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Trace_ELBO, autoguide
import math
from tqdm import tqdm

class DataLoader:
    def __init__(
        self,
        *args,
        batch_size: int,
        seed : int = 0,
        min_steps: int = 1,
        max_steps: int = None,
        n_epochs: int = None,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.args = args
        self.n = len(self.args[0])
        self.batch_size = batch_size
        self.drop_last = drop_last

        if max_steps is not None:
            n_steps = max_steps
        elif n_epochs is not None:
            n_steps = n_epochs * math.ceil(self.n / batch_size)
            n_steps = max(n_steps, min_steps)
        else:
            raise ValueError("Must set one of max_steps, n_epochs.")

        self.total_steps = n_steps
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        self.step = 0
        self.index = 0
        if self.shuffle:
            self.ordering = self.rng.permutation(self.n)
        else:
            self.ordering = np.arange(self.n)
        return self

    def __next__(self):
        if self.step < self.total_steps:
            if (self.index >= self.n) or ((self.index >= (self.n-self.batch_size)) and self.drop_last):
                self.index = 0
                if self.shuffle:
                    self.ordering = self.rng.permutation(self.n)

            idx = self.ordering[self.index : (self.index + self.batch_size)]
            self.index += self.batch_size
            self.step += 1
            curr = tuple(x[idx] for x in self.args)
            return curr
        else:
            raise StopIteration

    def __len__(self):
        return self.total_steps

    def just_passed_epoch(self):
        return self.index >= self.n

class BernoulliFactorModel:
    def __init__(self, n_rows:int, n_cols:int, batch_size:int, n_steps:int, seed:int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_steps = n_steps
        self.rng_key = PRNGKey(seed)
        self.seed = seed
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def model(self, 
              row_ids: Array, 
              col_ids: Array, 
              obs: Array = None):
        
        batch_size = row_ids.shape[0]

        row_embed_sigma = numpyro.sample("row_embed_sigma", dist.Gamma(1, 1))
        col_embed_sigma = numpyro.sample("col_embed_sigma", dist.Gamma(1, 1))

        row_embed = numpyro.sample("row_embed",
            dist.Normal(
                jnp.zeros((self.n_rows, self.n_dims)),
                row_embed_sigma,
            ).to_event(2),
        )

        col_embed = numpyro.sample("col_embed",
            dist.Normal(
                jnp.zeros((self.n_cols, self.n_dims)),
                col_embed_sigma,
            ).to_event(2),
        )

        # with numpyro.plate("plate", size=n_data, subsample_size=batch_size):
        with numpyro.plate("plate", size=self.n_data, subsample_size=batch_size):
            pred_logits = jnp.sum(row_embed[row_ids]*col_embed[col_ids], axis=-1)
            v = numpyro.sample("observation", dist.Bernoulli(logits=pred_logits), obs=obs)

        return(expit(pred_logits))
    
    def predict(self, row_ids:np.ndarray, col_ids:np.ndarray, prob_transform:bool=False):
        logits = jnp.sum(self.row_embed[row_ids]*self.col_embed[col_ids], axis=-1)
        if prob_transform:
            return(expit(logits))
        else:
            return(logits)

    def fit(self, 
            row_ids:np.ndarray, 
            col_ids:np.ndarray, 
            obs:np.ndarray, 
            n_dims:int,
            step_size:float=0.001, 
            progress_bar:bool=True):
        
        self.n_data = row_ids.shape[0]
        self.n_dims = n_dims
         
        loader = DataLoader(jnp.array(row_ids, dtype=int), 
                            jnp.array(col_ids, dtype=int), 
                            jnp.array(obs, dtype=float),
                            batch_size=self.batch_size, 
                            seed=self.seed, 
                            max_steps=self.n_steps, 
                            shuffle=True, 
                            drop_last=((self.n_data % self.batch_size)!=0))
        
        optimizer = numpyro.optim.Adam(step_size)
        guide = autoguide.AutoNormal(self.model)
        svi = numpyro.infer.SVI(self.model, guide, optimizer, Trace_ELBO())
        jit_update = jit(svi.update)

        r_idx, c_idx, y = next(iter(loader))
        svi_state = svi.init(self.rng_key, row_ids=r_idx, col_ids=c_idx, obs=y)

        losses = []
        for r_idx, c_idx, y in tqdm(loader, disable=(not progress_bar)):
            svi_state, loss = jit_update(svi_state, row_ids=r_idx, col_ids=c_idx, obs=y)    
            losses.append(loss.item())
        
        params = svi.get_params(svi_state)
        self.row_embed = params['row_embed_auto_loc']
        self.col_embed = params['col_embed_auto_loc']
        return(losses)
    

    def fit_via_cv(self, 
                   row_ids:np.ndarray, 
                   col_ids:np.ndarray, 
                   obs:np.ndarray, 
                   dim_list:list[int], 
                   n_folds:int,
                   step_size:float=0.001, 
                   progress_bar:bool=True):
        
        splits = np.array_split(self.rng.permutation(row_ids.shape[0]), n_folds)
        all_lls = []
        for n_dims in tqdm(dim_list, disable=(not progress_bar)):
            lls = []
            for i, test_idx in enumerate(splits):
                train_idx = np.setdiff1d(np.arange(row_ids.shape[0]), test_idx)
                self.fit(row_ids=row_ids[train_idx], col_ids=col_ids[train_idx], obs=obs[train_idx], n_dims=n_dims, step_size=step_size, progress_bar=False)
                logits = self.predict(row_ids[test_idx], col_ids[test_idx], prob_transform=False)
                lls.append(jnp.sum(dist.Bernoulli(logits=logits).log_prob(obs[test_idx])).item())
            all_lls.append(np.mean(lls))

        best_model_index = np.argmax(all_lls)
        best_dim = dim_list[best_model_index]

        losses = self.fit(row_ids=row_ids, col_ids=col_ids, obs=obs, n_dims=best_dim, step_size=step_size, progress_bar=True)

        return(all_lls, losses)
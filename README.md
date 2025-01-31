# The Causal Two Groups Model

This package implements algorithms for fitting nonparametric causal two-groups models. 


## Background

The causal two-groups model assumes that data is generated according to the following generative model:

```
y | x, h    ~  (1-h) f_0(y | x) + h f_1(y | x)
h | x, t=0  =  0
h | x, t=1  ~  Bernoulli( p(x) )
```
Here, x is a vector of covariates, y is a response value, t denotes whether or not the individual was prescribed a treatment, and h is a random variable describing whether or not the individual saw an effect from the treatment.


## Installation

To install this package, clone the git repo and run.

```
pip install -e .
```

## Fitting a causal two-groups model



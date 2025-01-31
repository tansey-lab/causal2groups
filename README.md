# The Causal Two Groups Model

This package implements algorithms for fitting nonparametric causal two-groups models. 


## Background

The causal two-groups model assumes that data is generated according to the following generative model:

```
y | x, h    ~  (1-h) f_0(y | x) + h f_1(y | x)
h | x, t=0  =  0
h | x, t=1  ~  Bernoulli( p(x) )
```
Here, x is a vector of covariates, y is an outcome value, t denotes whether or not the individual was prescribed a treatment, and h is a random variable describing whether or not the individual saw an effect from the treatment. p is an unknown function mapping covariates to [0,1]. f_0 and f_1 are conditional distributions over the outcome values. This package considers two ways of fitting this model.

The first approach is the additive causal two-groups model, which places the following form on f_0 and f_1:
```
y | x, h=0  =  mu_0(x) + epsilon
y | x, h=1  =  mu_1(x) + epsilon
epsilon     ~  g(epsilon)
```
Here, mu_0 and mu_1 are arbitrary functions and g is an arbitrary noise distribution. This model is identifiable, so our learned parameters here are meaningful. We model mu_0 and mu_1 via kernel ridge regression, and we approximate g using predictive recursion.

## Installation

To install this package, clone the git repo and run.

```
pip install -e .
```

## Fitting a causal two-groups model



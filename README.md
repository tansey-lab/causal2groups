# The Causal Two Groups Model

This package implements algorithms for fitting nonparametric causal two-groups models. 


## Background

The causal two-groups model assumes that data is generated according to the following generative model:

$$
\begin{aligned}
y | x, h    &~&  (1-h) f_0(y | x) + h f_1(y | x) \\
h | x, t=0  &=&  0 \\
h | x, t=1  &~&  \text{Bernoulli}( \pi(x) )
\end{aligned}
$$

Here, $x$ is a vector of covariates, $y$ is an outcome value, $t$ denotes whether or not the individual was prescribed a treatment, and $h$ is a random variable describing whether or not the individual saw a response from the treatment. $\pi$ is an unknown function mapping covariates to [0,1]. $f_0$ and $f_1$ are conditional distributions over the outcome values. This package considers two ways of fitting this model.

The first approach is the additive causal two-groups model, which places the following form on f_0 and f_1:

$$
\begin{aligned}
y | x, h=0  &=&  \mu_0(x) + \epsilon \\
y | x, h=1  &=&  \mu_1(x) + \epsilon \\
\epsilon     &\sim&  g(\epsilon)
\end{aligned}
$$

Here, $\mu_0$ and $\mu_1$ are arbitrary functions and $g$ is an arbitrary noise distribution. This model is identifiable, so our learned parameters here are meaningful. We model $\mu_0$ and $\mu_1$ via kernel ridge regression, we model $\pi$ using logistic regression with random Fourier features, and we approximate $g$ using predictive recursion. Given these estimates, we can perform selection of individuals based on the estimated posterior probability of $H=1 | X=x, Y=y$ while controlling the false discovery rate (FDR). Additionally, we can use our estimates to approximate $\mu_1(x) - \mu_0(x)$, the conditional average response effect (CARE), as well as $\frac{1}{\sum_i t_i} \sum_i t_i \pi(x_i)$, the expected responder population fraction (ERPF).

The second approach is to be completely non-parametric. In this case, $\pi$ and $f_1$ are not identifiable in general. However, we do know that $\pi$ must lie in an interval [$\pi^\star(x)$, 1], where

$$\pi^\star(x) = 1 - \min_y \frac{f_t(y | x)}{f_0(y|x)},$$

where $f_t$ is the treatment distribution, i.e. $p(Y=y | X=x, T=1)$. We model $f_0$ and $f_t$ using bootstrapped kernel density estimation, and we use the resulting estimands to numerically solve for $\pi^\star(x)$ for each individual $x$. Given $\pi^\star$, we can form a conservative posterior estimate of $H=1 | X=x, Y=y$ to select individuals while controlling FDR. We can also use the fact that $\pi$ lies in [$\pi^\star(x)$, 1] to give interval estimates of the CARE and the ERPF.

## Installation

To install this package, clone the git repo and run.

```
pip install -e .
```

## Fitting a causal two-groups model



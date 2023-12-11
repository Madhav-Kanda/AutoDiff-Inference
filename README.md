# AutoDiff-Inference ([Bijax](https://github.com/patel-zeel/bijax))
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

This repository contains code for implementing Automatic Differentiation Variational Inference (ADVI) and different variants of Laplace Approximation based on major research papers.

### Features:

- **ADVI Implementation** 
- **Laplace Approximation:** Implementation of Laplace Approximation for constrained variables, inspired by Automatic Differentiation Variational Inference (ADVI).


## Laplace Approximation (LA)

### Implementation

```python
  ## Creation of the dataset for Laplace Approximation
  data_dist = tfd.Bernoulli(probs=0.7)
  data = data_dist.sample(sample_shape=(100,), seed=jax.random.PRNGKey(3))
  prior_theta = [3.0, 5.0]     
```

```python
## Bernoulli likelihood function
def likelihood_fn(theta, data):
    return tfd.Bernoulli(probs=theta).log_prob(data).sum()

# For Posterior distribution
alpha = prior_theta[0] + data.sum()
beta = prior_theta[1] + len(data) - data.sum()
```
----------------------------------------------------

Normal Laplace Approximation
```python
## Using Identity bijector for normal Laplace Approximation
   la = LaplaceApproximation(
    prior=tfd.Beta(prior_theta[0], prior_theta[1]),
    bijector=tfp.bijectors.Identity(),                  
    likelihood=likelihood_fn)
```

```python
true_posterior = tfd.Beta(alpha, beta)      ## True posterior

fig = la.plot_approx_posterior(true_posterior=true_posterior)     

plt.xlim(-0.5,1.5)
plt.figure()
plt.savefig("plots/la_coin_toss.png")
```
<img width="500" alt="image" src="https://github.com/Madhav-Kanda/AutoDiff-Inference/assets/76394914/a808f463-2e2d-49a1-a122-813a3b0fd756">

-----------------------------------------------------
Autodiff- Laplace Appoximation
```python
## Using Sigmoid bijector for constrained Laplace Approximation
la_cov = LaplaceApproximation(
    prior=tfd.Beta(prior_theta[0], prior_theta[1]),
    bijector=tfp.bijectors.Sigmoid(),           
    likelihood=likelihood_fn)
```
```python
true_posterior = tfd.Beta(alpha, beta)

fig_cov = la_cov.plot_approx_posterior(true_posterior=true_posterior)
plt.figure()
plt.savefig("plots/la_cov_coin_toss.png")

fig = la_cov.plot_log_approx_posterior(true_posterior=true_posterior)
plt.savefig("plots/log_la_cov_coin_toss.png")
```
<divr>
<img width="500" alt="image" src="https://github.com/Madhav-Kanda/AutoDiff-Inference/assets/76394914/14390528-dfc3-4b6c-9c41-fa86b59b586e">
</div>

<br><br>
In addition to the implemented library for Laplace approximation, you'll find two additional notebooks showcasing diagonal Laplace approximation and low-rank Laplace approximation.

--------------------------------------------

## Automatic Differentiation Variational Inference (ADVI)
### Implementation

```python
tfd = tfp.distributions
data_dist = tfd.Bernoulli(probs=0.7)
data = data_dist.sample(sample_shape=(100,), seed=jax.random.PRNGKey(3))
prior_theta = [3.0, 5.0]

```

```python
def likelihood_fn(theta, data):
    return tfd.Bernoulli(probs=theta).log_prob(data).sum()
```

```python
advi = ADVI(
    prior=tfd.Beta(prior_theta[0], prior_theta[1]),
    bijector=tfp.bijectors.NormalCDF(),
    likelihood=likelihood_fn,
)
appx_post = advi.approx_posterior(data)
```
<img width="500" alt="image" src="https://github.com/Madhav-Kanda/AutoDiff-Inference/assets/76394914/d3d00ebc-9655-4d04-a5b8-2096dc759059">


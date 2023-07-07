from advi import ADVI
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


tfd = tfp.distributions
data_dist = tfd.Bernoulli(probs=0.7)
data = data_dist.sample(sample_shape=(100,), seed=jax.random.PRNGKey(3))
prior_theta = [3.0, 5.0]


def likelihood_fn(theta, data):
    return tfd.Bernoulli(probs=theta).log_prob(data).sum()


advi = ADVI(
    prior=tfd.Beta(prior_theta[0], prior_theta[1]),
    bijector=tfp.bijectors.NormalCDF(),
    likelihood=likelihood_fn,
)
appx_post = advi.approx_posterior(data)

alpha = prior_theta[0] + data.sum()
beta = prior_theta[1] + len(data) - data.sum()
true_posterior = tfd.Beta(alpha, beta)
fig = advi.plot_approx_posterior(true_posterior=true_posterior)
plt.savefig("advi_coin_toss.png")

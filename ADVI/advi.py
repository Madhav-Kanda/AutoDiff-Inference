import logging
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import seaborn as sns
import tensorflow_probability.substrates.jax as tfp
from tqdm import trange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())
tfd = tfp.distributions


class ADVI:
    def __init__(
        self, prior, bijector=tfp.bijectors.Identity(), likelihood=None
    ) -> None:
        self.prior = prior
        self.bijector = bijector
        self.likelihood = likelihood

    # ELBO Loss function
    def elbo(self, mu_sigma, data, seed, n_samples):

        bijector = self.bijector
        prior = self.prior

        q = tfd.MultivariateNormalDiag(
            loc=mu_sigma["mu"], scale_diag=jnp.exp(mu_sigma["sigma"])
        )
        transformed_prior = tfd.TransformedDistribution(
            distribution=prior, bijector=tfp.bijectors.Invert(bijector)
        )

        theta = q.sample(seed=jax.random.PRNGKey(seed), sample_shape=(n_samples,))

        expected_q = q.log_prob(theta).mean()
        expected_log_prior = transformed_prior.log_prob(theta).mean()
        expected_log_likelihood = jax.vmap(self.likelihood, in_axes=(0, None))(
            bijector(theta), data
        ).mean()

        return expected_q - expected_log_prior - expected_log_likelihood

    def approx_posterior(
        self,
        data,
        mu_sigma=None,
        max_iter=500,
        lr=0.03,
        n_samples=1000,
        seed=jax.random.PRNGKey(0),
    ):
        key1, key2 = jax.random.split(seed, 2)
        d = data.shape[1] if len(data.shape) > 1 else 1
        if mu_sigma is None:
            mu_sigma = {
                "mu": jax.random.uniform(shape=(d,), key=key1),
                "sigma": jnp.log(jax.random.uniform(shape=(d,), key=key2)),
            }

        value_and_grad_fn = jax.jit(jax.value_and_grad(self.elbo), static_argnums=[3])
        optimizer = optax.adam(learning_rate=lr)
        state = optimizer.init(mu_sigma)

        losses = []
        for _ in trange(max_iter):
            value, grad = value_and_grad_fn(mu_sigma, data, seed=_, n_samples=n_samples)
            losses.append(value)
            update, state = optimizer.update(grad, state)
            mu_sigma = optax.apply_updates(mu_sigma, update)
        self.losses = losses
        print(mu_sigma["mu"], jnp.exp(mu_sigma["sigma"]))
        print(f"Final loss = {losses[-1]}")

        self.q = self.bijector(
            tfd.Normal(loc=mu_sigma["mu"], scale=jnp.exp(mu_sigma["sigma"]))
        )

        return self.q

    def plot_loss(self):
        if self.losses is None:
            raise ValueError("Run approx_posterior first")
        # plt.figure(figsize=(16,12))
        plt.plot(self.losses)
        plt.legend()
        plt.title("Loss vs iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        sns.despine()
        return 

    def plot_approx_posterior(self, true_posterior=None):
        if self.q is None:
            raise ValueError("Run approx_posterior first")
        x = jnp.linspace(-5, 5, 1000)
        fig = plt.figure()
        if true_posterior is not None:
            plt.plot(x, true_posterior.prob(x), label="true_posterior", color="orange")
        plt.plot(
            x,
            self.q.prob(x),
            label=r"$q(\theta)$ ~ Normal",
            linestyle="dashed",
            color="k",
        )
        plt.title("Approximate Posterior")  # with n_samples = '+str(n_samples))
        plt.xlabel("x")
        plt.ylabel("pdf(x)")
        plt.legend()
        sns.despine()
        return fig

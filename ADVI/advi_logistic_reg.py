from advi import ADVI
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import make_classification
import numpy as np

tfd = tfp.distributions

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

data = jnp.c_[X, y]


def log_likelihood(theta, data):
    X, y = data[:, :-1], data[:, -1]
    l = tfd.Bernoulli(probs=jax.nn.sigmoid(X @ theta[:-1] + theta[-1]))
    return l.log_prob(y).sum()


b = 1.0
advi = ADVI(
    prior=tfd.MultivariateNormalDiag(loc=[0.0, 0.0, 0.0], scale_diag=[b, b, b]),

    # bijector=tfp.bijectors.NormalCDF(),
    likelihood=log_likelihood,
)
appx_post = advi.approx_posterior(data)
samples = appx_post.sample(sample_shape=(1000,), seed=jax.random.PRNGKey(0))
s = sns.jointplot(x=samples[:, 0], y=samples[:, 1], kind="kde", bw_adjust=2)
s.set_axis_labels(r"$\theta_0$", r"$\theta_1$")
s.fig.suptitle("ADVI Logistic Regression Approximate posterior")
s.fig.tight_layout()
os.makedirs("plots/", exist_ok=True)
s.fig.savefig("plots/advi_log_reg.png")
plt.figure()
advi.plot_loss()

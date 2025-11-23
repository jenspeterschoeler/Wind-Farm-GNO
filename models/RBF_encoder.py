"""RBF encoder module for graph neural networks taken from the implementation of a conventional neural network intended to estimate the position of atoms in a material [1], the code implementation is inspired by [2].

[1]: Behler, J., & Parrinello, M. (2007). Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces. Physical Review Letters, 98(14), 146401. https://doi.org/10.1103/PhysRevLett.98.146401
[2]: https://github.com/microsoft/AI2BMD/blob/main/src/ViSNet/model/utils.py"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class RBFEncoder(nn.Module):
    """Currently only supports a Gaussian basis and only Cosine cutoff function"""

    num_kernels: int  # Number of RBF basis functions
    d_min: float
    d_max: float
    learnable: bool = True

    def setup(self):
        # Define centers (mu) and scaling factors (beta)
        mu = jnp.linspace(self.d_min, self.d_max, self.num_kernels)
        beta = jnp.ones(self.num_kernels) * (
            self.num_kernels / (self.d_max - self.d_min)
        )

        # Make learnable if required
        self.mu = self.param("mu", lambda rng: mu) if self.learnable else mu
        self.beta = self.param("beta", lambda rng: beta) if self.learnable else beta

    def CosineCutOff(self, distances):
        """Smooth cosine cutoff function"""
        cutoff = 0.5 * (jnp.cos(jnp.pi * distances / self.d_max) + 1.0)
        cutoff = jnp.where(distances < self.d_max, cutoff, 0.0)
        return cutoff

    def __call__(self, distances):
        """Compute the RBF features for input distances"""
        return self.CosineCutOff(distances)[..., None] * jnp.exp(
            -self.beta * (distances[..., None] - self.mu) ** 2
        )


if __name__ == "__main__":
    import os
    import sys

    import matplotlib.pyplot as plt

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.plotting import matplotlib_set_rcparams

    matplotlib_set_rcparams(publication="paper")

    ### Setup the RBF layer and encode some distances
    d_min = -1.0
    d_max = 1.0
    distances = jnp.linspace(d_min, d_max, 100)

    key = jax.random.PRNGKey(0)
    num_kernels = 9
    rbf_layer = RBFEncoder(
        num_kernels=num_kernels, d_min=d_min, d_max=d_max, learnable=True
    )
    params = rbf_layer.init(key, distances)  # Initialize parameters

    encoded_distances = rbf_layer.apply(params, distances)
    mu = params["params"]["mu"]
    beta = params["params"]["beta"]

    ### Make a plot of the RBF functions
    # Make matching colors and linstyles for the symmetrical kernels i.e. 1 and 9, 2 and 8, etc.
    num_colors = (num_kernels + 1) // 2
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(num_colors)]
    line_styles = ["-", "--"]
    if num_kernels % 2 == 0:
        colors = colors + colors[::-1]
        line_styles = [line_styles[0]] * (num_kernels // 2) + [line_styles[1]] * (
            num_kernels // 2
        )
    elif num_kernels % 2 == 1:
        colors = colors + colors[-2::-1]
        line_styles = [line_styles[0]] * (1 + (num_kernels // 2)) + [line_styles[1]] * (
            num_kernels // 2
        )

    plt.figure()
    for i in range(rbf_layer.num_kernels):
        plt.plot(
            distances,
            encoded_distances[:, i],
            label=r"$\mu_\mathrm{RBF}^{("
            f"{i+1}"
            + r")}$"
            + f"={mu[i]:.2f},",  # + r"$\beta_\mathrm{RBF}$="f"{beta[i]:.2f}",
            color=colors[i],
            linestyle=line_styles[i],
        )
    plt.xlabel(r"$d_{ij}/d_c$ [-]")
    plt.ylabel(r"$\varphi_\mathrm{RBF}(d_{ij})$ [-]")
    plt.legend(bbox_to_anchor=(0.45, 1.65), loc="upper center", ncol=3)
    plt.savefig(os.path.abspath("./assets/RBF_encoder.pdf"), bbox_inches="tight")

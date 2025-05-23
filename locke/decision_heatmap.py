import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
d_prime = 2.0       # sensitivity: distance between means
sigma = 1.0         # equal variances
mu_L = -d_prime/2
mu_R = +d_prime/2

# Grids: now meshgrid over (x_vals, priors) so rows=priors, cols=x
x_vals = np.linspace(-4, +4, 400)            # stimulus tilt axis (x-axis)
priors = np.linspace(0.25, 0.75, 300)        # prior axis  (y-axis)
X_grid, P_grid = np.meshgrid(x_vals, priors)  # X_grid.shape = (len(priors), len(x_vals))

# Likelihoods
p_x_sL = norm.pdf(X_grid, loc=mu_L, scale=sigma)
p_x_sR = norm.pdf(X_grid, loc=mu_R, scale=sigma)

# Priors
P_sL = P_grid
P_sR = 1 - P_sL

# Unnormalized posteriors
uL = p_x_sL * P_sL
uR = p_x_sR * P_sR

# Normalize
evidence = uL + uR
post_L = uL / evidence
post_R = uR / evidence

# Build RGB image: shape = (n_priors, n_x, 3)
img = np.zeros((priors.size, x_vals.size, 3))
img[..., 0] = post_L    # red channel = P(s_L|x)
img[..., 1] = post_R    # green = P(s_R|x)

# Plot with x-axis = x_vals, y-axis = priors
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    img,
    origin='lower',
    extent=[x_vals.min(), x_vals.max(), priors.min(), priors.max()],
    aspect='auto'
)
ax.set_xlabel("Internal measurement $x$ (tilt)")
ax.set_ylabel("Prior $P(s_L)$")
ax.set_title("Bayesian decision heatmap\n(red=“left”, green=“right”)")
plt.tight_layout()
plt.show()

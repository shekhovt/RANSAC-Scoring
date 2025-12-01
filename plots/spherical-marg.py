# %%
# prompt: Given a 1D function $h(r)$, compute: integral of $h(\|x\|)$ over R^d using a finite sum over the radius $r$ and the analytical formula for the hyper-area of d-1 dimensional sphere 
from .spherical_density import *

# %%
import matplotlib.pyplot as plt
def h1(r):
   return np.exp(-r**2)

def h2(r):
  return np.exp(-np.abs(r))

def h3(r):
  return r<=3

d = 4
r_min = 0
r_max = 8
num_bins = 1000

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5)) # Create a figure with two subplots

plabels = ['$h(\rho) = e^{-\rho^2}$', '$h(\rho) = e^{-|\rho|}$', '$h(\rho) = [\rho{\leq}3]$']

for i, h in enumerate([h1, h2, h3]):
  r, m = spherical_1D_marginal(h, d, r_min, r_max, num_bins)
  r, p = spherical_1D_ray(h, d, r_min, r_max, num_bins)
  fg = fit_gaussian_density(r, m)
  q  = density_of_r(h,r,d)
  ax = axes[i] # Select the current axis
  # ax.plot(r, p, '-r', label='Radial profile ' + plabels[i])
  me = 100
  ms = 5
  ax.plot(r, p, '-rs', label='Radial profile $h(\rho)$', markevery = me, markersize=ms)
  ax.plot(r, m, '--bx', label='1D Marginal Density', markevery = me + 20, markersize=ms)
  if i >0:
    ax.plot(r, fg, ':g', label='Gaussian Approx', markevery = me + 30, markersize=ms)

  ax.plot(r, q, '-k', label='Density of $\rho$', markevery = me + 40, markersize=ms, linewidth=0.5)
  # ax.set_yscale('log')
  ax.set_xlabel('Radius (\rho)')
  ax.set_ylabel('Density')
  ax.set_title(plabels[i]) # Add a specific title for each plot
  ax.legend()
  ax.grid(True)
  ax.set_xlim(left=0)

# # Get the y-axis limits from the first plot
# y_min, y_max = axes[0].get_ylim()
# # Align the y-axis limits of the other plots to the first plot
# for i in range(1, 3):
#     axes[i].set_ylim(y_min, y_max)


plt.tight_layout() # Adjust layout to prevent overlap

plt.savefig('spherical-marg.pdf', bbox_inches='tight', pad_inches=0.0)

plt.show()
# %%

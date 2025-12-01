 #%%
# prompt: Given a 1D function $h(r)$, compute: integral of $h(\|x\|)$ over R^d using a finite sum over the radius $r$ and the analytical formula for the hyper-area of d-1 dimensional sphere 

import numpy as np
from scipy.special import gamma

def hypersphere_area(d, R):
  """Calculates the surface area of a d-dimensional sphere of radius R."""
  return 2 * np.pi**(d / 2) / gamma(d / 2) * R**(d - 1)

def integrate_spherical(h, d, r_min, r_max, num_bins, **kwargs):
  """
  Computes the integral of h(||x||) over R^d using a finite sum
  over the radius and the analytical formula for the hyper-area.

  Args:
    h: A function that takes a single scalar argument (the radius).
    d: The dimension of the space.
    r_min: The minimum radius for integration.
    r_max: The maximum radius for integration.
    num_bins: The number of bins to use for the radial integration.

  Returns:
    The approximate integral of h(||x||) over R^d.
  """
  radii = np.linspace(r_min, r_max, num_bins + 1)
  delta_r = (r_max - r_min) / num_bins

  # Use the midpoints of the bins for the radius
  r_mids = (radii[:-1] + radii[1:]) / 2

  # Calculate the volume elements for all midpoints
  volume_elements = hypersphere_area(d, r_mids) * delta_r

  # Calculate h(r_mid) for all midpoints and perform the element-wise multiplication
  integral = np.sum(h(r_mids, **kwargs) * volume_elements)

  return integral

# Example usage:
# Let's integrate a simple function, say h(r) = r^2, in 3 dimensions
# from r=0 to r=2.
def h_example(r, **kwargs):
  return r**2

d = 3
r_min = 0
r_max = 2
num_bins = 1000

integral_result = integrate_spherical(h_example, d, r_min, r_max, num_bins)
print(f"Approximate integral: {integral_result}")

# For h(r) = r^2 in 3D from 0 to 2, the analytical integral is:
# Integral[r^2 * (4*pi*r^2) dr] from 0 to 2
# Integral[4*pi*r^4 dr] from 0 to 2
# 4*pi * [r^5 / 5] from 0 to 2
# 4*pi * (32 / 5 - 0) = 128*pi / 5
analytical_result = 128 * np.pi / 5
print(f"Analytical integral: {analytical_result}")

# %%
def spherical_1D_marginal(h, d, r_min, r_max, num_bins):
  # Computes the 1D marginal density of a spherical density with profile h(r)
  z = np.linspace(r_min, r_max, num_bins + 1)
  delta_r = (r_max - r_min) / num_bins
  zz = (z[:-1] + z[1:]) / 2
  m = np.zeros(len(zz))
  # V = integrate_spherical(h,d,r_min,r_max,num_bins)
  def hz(r, z = 0):
    return h((z**2 + r **2)**0.5)
  for i in range(num_bins):
    m[i] = integrate_spherical(hz, d-1, r_min, r_max, num_bins, z = zz[i])
  m = m / m.sum() # renormalize as discrete
  return z,m

def spherical_1D_ray(h, d, r_min, r_max, num_bins):
  z = np.linspace(r_min, r_max, num_bins+1)
  zz = (z[:-1] + z[1:]) / 2
  p = h(zz)
  p = p / p.sum() # renormalize as discrete
  return zz,p

#%%

import numpy as np
from scipy.optimize import curve_fit

def zero_mean_gaussian(x, sigma, A):
    """1D zero-mean Gaussian with scale sigma and amplitude A"""
    return A * np.exp(-x**2 / (2 * sigma**2))

def fit_zero_mean_gaussian(x, y):
    """
    Fit a zero-mean Gaussian density A * exp(-x^2 / (2*sigma^2)) to data.
    
    Parameters:
        x (array-like): input x values
        y (array-like): corresponding y values (density estimates)
    
    Returns:
        sigma (float): estimated standard deviation
        A (float): estimated amplitude
        y_fit (np.array): fitted values at x
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Initial guesses: sigma from weighted variance, A from max
    sigma0 = np.sqrt(np.sum(x**2 * y) / np.sum(y))
    A0 = np.max(y)
    
    popt, _ = curve_fit(zero_mean_gaussian, x, y, p0=[sigma0, A0])
    sigma_est, A_est = popt
    
    return sigma_est, A_est

def fit_gaussian_density(r, m):
  mask = r < 4
  sigma, A = fit_zero_mean_gaussian(r[mask], m[mask])
  q = zero_mean_gaussian(r, sigma, A)
  q = q / q.sum()
  return q

def density_of_r(h, r, d):
  q = h(r) * hypersphere_area(d, r)
  q = q /q.sum()
  return q
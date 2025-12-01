# %%
import matplotlib.pyplot as plt
import numpy as np
# plot max(1- r^2/\tau^2, 0) for r in [0,2]

def smax(a,b):
    return np.log(np.exp(a) + np.exp(b))

plt.figure()

r = np.linspace(0, 2, 100)
tau = 1
y = np.maximum(1 - r**2 / tau**2, 0)
plt.plot(r, y, label='MSAC', color='blue')
for k in [1, 2, 4]:
    # sigma = 1/ 4
    # k = tau**2 / (2 * sigma**2)
    y = smax(k*(1 - r**2 / tau**2), 0)
    # y = smax((tau**2 - r**2) / (2*sigma**2), 0)
    y = y / y[0]  # Normalize to max value
    plt.plot(r, y, label=f'k={k}', linestyle='--')

plt.xlabel('r')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.ylim(0, 1)
plt.show()
# %%

#%%
# translation rotation# %%
import os
import sys
print(__name__)
if __name__ == "__main__":
    __name__ = 'score_learn.manifold_H.py'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
else:
    __run__ = False

from .drawing import *

# %%
plt.figure()
import numpy as np

x = np.linspace(0,0.75,100)
y = 1/(0.8-x) - 0.2
plt.plot(x,y)

plt.gca().axis('off')
plt.draw()
plt.savefig('fig/manifold_H.pdf')
plt.show()


# %%

import matplotlib.pyplot as plt
import matplotlib

from .tools import *


SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the figure title

prop_cycle = plt.rcParams['axes.prop_cycle']
cc = prop_cycle.by_key()['color']

import matplotlib.colors as mcolors
cc1 =  list(mcolors.TABLEAU_COLORS)
cc1 = cc1 + cc1
cc1 = cc1 + ['red','green','blue']
cc = cc1

markers = 'ov^<>sp*ox+ds123'

def savefig(outf):
    print(outf)
    force_path(outf)
    plt.savefig(outf, bbox_inches='tight', pad_inches=0.0)
    
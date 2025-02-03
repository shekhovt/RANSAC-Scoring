# %%
from score_learn.load_data import *
from score_learn import *
from types import SimpleNamespace
import os
import scipy.stats
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#!%matplotlib inline

# src = '/tmp/RANSAC/united_states_capitol/'; sqrt = True
# src = '/tmp/RANSAC/british_museum/'; sqrt = True
# src = '/tmp/RANSAC/st_peters_square1/'; sqrt = True
# src = '/tmp/RANSAC/st_peters_square_RootSIFT/'; sqrt = True
src = '/tmp/RANSAC/KITTI/train/'
sqrt = True
F_matrix = True
# residual_dataset = ResidualData(src, padding=True, sqrt=sqrt) # padding the residuals with infs
residual_dataset = ResidualData(
    src, padding=True, sqrt=sqrt, size=10000)  # padding the
data_loader = torch.utils.data.DataLoader(
    residual_dataset, batch_size=16, num_workers=0, shuffle=True)

# _________VALIDATION______________
# %% VALIDATINO
max_distance = 10
N_bins = 200  # filter to select models to evaluate
methods = []
# pth = "./models/KITTI/"
pth = "./models/"
files = os.listdir(pth)
# files = ['monotone_tau=10.npz']
# files = ['monotone_tau=10_bins=2000.npz']
# files = ['msac_tau=3.0.npz']
for file in files:
    if file.endswith(".pkl"):
        f = os.path.join(pth, file)
        W = torch.load(f)
        print(file)
        if hasattr(W,'o'):
            print(W.o)
    # if file.endswith(".npz"):
    #     f = os.path.join(pth, file)
    #     M = np.load(f)
    #     M.w = torch.tensor(M['weight']).cuda()
    #     if 'max_distance' in M.keys():
    #         M.max_distance = M['max_distance'].item()
    #     else:
    #         M.max_distance = M['tau'].item()
    #     # if np.abs(M.max_distance-max_distance)>1e-5 or len(M.w)!= N_bins:
    #         # print('skipping' + f + ',\t max_distance=' + str(M.max_distance))
    #         # continue
    #     M.name = file.replace('.npz', '').replace('magsac_', 'magsac   ').replace(
    #         'msac_', 'msac     ').replace('_', ' ')
    #     print(M.name + ',\t max_distance=' + str(M.max_distance))
    #     methods.append(M)

# %%

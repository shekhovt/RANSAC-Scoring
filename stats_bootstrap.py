import os
import sys
if __name__ == "__main__":
    __name__ = 'score_learn.stats_bootstrap.py'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
else:
    __run__ = False


import itertools
import warnings

from .load_data import *
from .metrics import *


# dataset_info = PhotoTourismRootSIFT
# dataset_info = PhotoTourismSPSG
dataset_info = KITTI
test_set = dataset_info.test
res_root = f'results/{dataset_info.name}/'
results_path = res_root + f'polish=all/'

stats = [np.median, AUC_10]
# selected = ['GaU', 'RANSAC', 'MSAC', 'Oracle']
selected = ['GaU', 'MSAC', 'MAGSAC++', 'RANSAC', 'ML   gamma=10.0', 'Oracle']
# selected = ['MAGSAC++', 'ML   gamma=10.0', 'Oracle']
polishes = ['GaU', 'TRUNCATED', 'TRUNCATED_LE_ZACH',0]
# polishes = [0]
err_name = dict(running_r='$e_R$', running_t='$e_t$', running_e='Pose Errror $e$')
err = 'running_e'
confidence = 0.95
nn_resamples = {AUC_10: 10000, np.median: 10000}
joint = False
atsamples = 4000
table_file = results_path + f'testr-{atsamples}-table.pkl'

def at_samples(polish):
    if polish == 0:
        return atsamples // mbatch_size - 1
    else:
        return atsamples // (mbatch_size*5)
        
mbatch_size = 100

def load_running(folder, polish):
    results_path = res_root + f'polish={polish}/'
    results_running_file = results_path + f'{folder}test_results_running.pkl'
    res = load_object(results_running_file)
    if 'methods' in res:
        methods = res['methods']
        files = res['files']
    else:
        methods = res
    methods = [m for m in methods if m.name in selected]
    return methods

mm = dict((mp, []) for mp in itertools.product(selected, polishes))

res = dict((f, dict()) for f in test_set)

for folder in test_set:
    methods = load_running(folder + '/', 'all')
    for m in methods:
        res[folder][m.name] = m


def mp_name(M,p):
    if p == 0:
        return M.name
    else:
        return M.name + '-' + p

maxl = np.max([len(mp_name(M,p)) for M in methods for p in polishes])


res_list = []

for stat in stats:
    n_resamples = nn_resamples[stat]
    print(f'____{"Mean" if not joint else "Total"} {stat.__name__}_at {atsamples} samples____')
    for i, M in enumerate(methods):
        for k, polish in enumerate(polishes):
            if polish != 0 and M.name not in ['GaU', 'Oracle']:
                continue
            print(mp_name(M,polish).ljust(maxl), end=': ')
            for err_key in ['running_e', 'running_r', 'running_t']:
                v = []
                d = []
                for f in test_set:
                    slice = at_samples(polish)
                    data = res[f][M.name].res[polish][err_key][0]
                    if len(data) > 0:
                        # import pdb; pdb.set_trace()
                        data = data[:, slice]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            br = scipy.stats.bootstrap((data,), stat, confidence_level=confidence, method='BCa', n_resamples=n_resamples)
                        ci = br.confidence_interval
                        v += [stat(data)]
                        d += [(ci.high - ci.low)/2]
                    else:
                        v += [np.nan]
                        d += [0]
                v = np.array(v).mean()
                d = np.array(d).mean()
                formatted = format_std(v, d)
                k = err_key.replace('best_', '')
                print(f'\t ({k}): {formatted}', end='')
                rec = dict(method=M.name, stat=stat.__name__, joint=joint, err_key = err_key, val = formatted, polish = polish)
                res_list += [rec]
            print('')

print(table_file)
force_path(table_file)
save_object(table_file, res_list)

    
if False:
    # _______________________ ANALYSIS______________________________________________
    # Statistics:
    # 1 median of e,r,t per dataset, averaged
    # 2 AUC of e,r,t per dataset, averaged
    # same for flat data

    # stats = [AUC_10]

    # joints = [True, False]
    joints = [False]

    maxl = 0
    for M in methods:
        maxl = max(maxl, len(M.name))
    confidence = 0.95
    nn_resamples = {AUC_10: 10, np.median: 10000}

    res_list = []

    print(f'____{dataset_info.name}_polish={polish}___')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for joint in joints:
            for stat in stats:
                n_resamples = nn_resamples[stat]
                print(
                    f'____{"Mean" if not joint else "Total"} {stat.__name__}_____')
                for M in methods:
                    print(M.name.ljust(maxl), end=': ')
                    for err_key in ['best_e', 'best_r', 'best_t']:
                        Data = getattr(M, err_key)
                        # print(f'______{stat.__name__} {err_key}_____')
                        if joint:
                            data = np.concatenate(Data, axis=0)
                            res = scipy.stats.bootstrap(
                                (data,), stat, confidence_level=confidence, method='BCa', n_resamples=n_resamples)
                            ci = res.confidence_interval
                            v = stat(data)
                            d = (ci.high - ci.low)/2
                        else:
                            v = []
                            d = []
                            for data in Data:
                                res = scipy.stats.bootstrap(
                                    (data,), stat, confidence_level=confidence, method='BCa', n_resamples=n_resamples)
                                ci = res.confidence_interval
                                v += [stat(data)]
                                d += [(ci.high - ci.low)/2]
                            v = np.array(v).mean()
                            d = np.array(d).mean()

                        formatted = format_std(v, d)
                        k = err_key.replace('best_', '')
                        print(f'\t ({k}): {formatted}', end='')
                        rec = dict(method=M.name, stat=stat.__name__, joint=joint, err_key = err_key, val = formatted)
                        res_list += [rec]
                    print('')

    force_path(table_file)
    save_object(table_file, res_list)

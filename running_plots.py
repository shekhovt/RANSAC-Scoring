# %%
import os
import sys
if __name__ == "__main__":
    __name__ = 'score_learn.running_plots.py'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
else:
    __run__ = False

import pandas as pd
import itertools

from .load_data import *
from .drawing import *

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


dataset_info = PhotoTourismRootSIFT
# dataset_info = PhotoTourismSPSG
# dataset_info = ETH3D
# dataset_info = LAMAR
# dataset_info = KITTI
# err = 'running_r'
err = 'running_e'
selected = ['GaU', 'MSAC', 'MAGSAC++', 'RANSAC', 'ML   gamma=10.0', 'Oracle', 'GT']
#polishes = [0, 'GaU', 'TRUNCATED', 'TRUNCATED_LE_ZACH']
polishes = [0] 

res_root = f'results/{dataset_info.name}/'
out = res_root + f'results.txt'


# polishes = [0, 'GaU', 'TRUNCATED']
# polishes = [0]

# dataset_info = PhotoTourismSPSG
# dataset_info = PhotoTourismRootSIFT
# res_root = f'results/{dataset_info.name}/'
# selected = ['GaU', 'RANSAC', 'Oracle']

# folder = dataset_info.test[7] + '/'
# folder = 'sagrada_familia/'
# folder = 'united_states_capitol/'
# folder = 'st_pauls_cathedral/'

# folder = ''


mbatch_size = 100
atsamples = 4000
def at_samples(polish):
    if polish == 0:
        return atsamples // mbatch_size - 1
    else:
        return atsamples // (mbatch_size*5)

def load_selected(polish):
    results_path = res_root + f'polish={polish}/'
    results_file = results_path + f'test_results.pkl'
    res = load_object(results_file)
    if 'methods' in res:
        methods = res['methods']
        files = res['files']
    else:
        methods = res
    methods = [m for m in methods if m.name in selected]
    return methods


def load_running(folder, polish):
    results_path = res_root + f'polish={polish}/'
    results_running_file = results_path + f'{folder}test_results_running.pkl'
    # print('loaded:', results_running_file)
    res = load_object(results_running_file)
    if 'methods' in res:
        methods = res['methods']
        files = res['files']
    else:
        methods = res
    # methods = [m for m in methods if m.name in selected]
    return methods

# %%

# mark = 'ox+ds<>^'
xrange = 5
Nbins = 50

if False:
    err = 'best_r'
    f = plt.figure()
    for polish in polishes:
        methods = load_running(polish)
        for i,M in enumerate(methods):
            Data = getattr(M,err)
            v = []
            for data in Data:
                v += [np.median(data)]
            mean_med = np.array(v).mean()
            data = np.concatenate(Data, axis=0)
            y, bine = np.histogram(data, Nbins, range=(0,xrange), density=False)
            med = np.median(data)
            name = M.name.replace('gamma=30.0', '').replace('MAGSAC', 'M').replace('pi=30.0','')
            print(name + f'\t med:{med:3.2f} | mmed: {mean_med:3.2f}')
            if polish == 0:
                style = '-'
                alpha= 1
            else:
                style = '--'
                name += '+LO'
                alpha=0.5
            plt.plot(bine[:-1], y, style, color = cc[i], label=name,alpha=alpha)
            # plt.hist(M.best_e)
    plt.ylabel(f'Number of scenes with error ±{xrange/ Nbins/2}')
    plt.xlabel('Error value')
    plt.legend()
    plt.gcf().tight_layout()
    plt.draw()
    plt.savefig(res_root + f'distribuiotn_{err}.pdf', bbox_inches="tight", format='pdf', transparent=True)
    plt.show()
    plt.close(f)

    # statistics of the gap
    xrange = 10
    methods = load_selected(0)
    f = plt.figure()
    data1 = np.concatenate(getattr(methods[0],err), axis=0)
    data0 = np.concatenate(getattr(methods[-1],err), axis=0)
    gap = data1 - data0
    x = np.sort(gap)[::-1]
    y = np.arange(len(x))
    # plt.plot(x, y)
    # plt.xlim(0.5, 20)
    # plt.ylim(0, 20000)
    # y, bine = np.histogram(gap, Nbins, range=(0, range), density=False)
    # plt.plot(bine[:-1], y, 'g', label='GaU-Oracle gap')
    # plt.bar(bine[:-1], y, label='GaU-Oracle gap')
    plt.hist(gap, Nbins, range=(0, xrange), density=False, edgecolor="k")
    plt.xlabel('Error Gap to the Oracle: Err(GaU)-Err(Oracle)')
    plt.ylabel('Number of scenes with gap' +f' ±{xrange/ Nbins/2}')
    plt.xlim(0)
    # plt.xlabel('g')
    # plt.legend()
    # plt.title('Error Gap to the Oracle: Err(GaU)-Err(Oracle)')
    plt.show()
    plt.close(f)

# %%
# polishes = [0, 'GaU', 'TRUNCATED_LE_ZACH']
# polishes = [0, 'TRUNCATED', 'GaU']
# polishes = [0, 'GaU', 'MSAC']
#polishes = ['GaU']

err_name = dict(running_r='$e_R$', running_t='$e_t$', running_e='Pose Error $e$')

#selected = ['GaU', 'MSAC', 'MAGSAC++', 'RANSAC', 'ML   gamma=10.0', 'Oracle', 'GT']
# polishes = [0, 'GaU', 'TRIVIAL', 'TRUNCATED', 'HUBER', 'CAUCHY', 'TRUNCATED_LE_ZACH']

# polishes = [0, 'GaU', 'TRUNCATED', 'CAUCHY', 'TRUNCATED_LE_ZACH']
# polishes = [0, 'GaU', 'TRUNCATED', 'TRUNCATED_LE_ZACH']

mm = dict((mp, []) for mp in itertools.product(selected, polishes))
mh = dict((mp, []) for mp in itertools.product(selected, polishes))

for folder in dataset_info.test + ['all']:
    if folder != 'all':
        try:
            if len(polishes) > 0:
                methods = load_running(folder +'/', 'all')
            else:
                assert(len(polishes) == 1 and polishes[0] == 0)
                methods = load_running(folder + '/', '0')
            # print(folder)
        except:
            print('no ', folder)
            continue
    name_to_method = {m.name:m for m in methods}
    f1 = plt.figure(figsize= (6,5))
    # f2 = plt.figure(figsize= (6,5))
    f2 = plt.figure()
    plt.figure(f1)
    i = -1
    for k,polish in enumerate(polishes):
        for Mname in selected:
            if Mname not in name_to_method:
                continue
            M = name_to_method[Mname]
            if Mname == 'GT' and polish == 0:
                continue
            if polish != 0:
                if Mname not in ['GaU', 'Oracle', 'GT']:
                    continue
                if Mname in ['Oracle', 'GT'] and polish != 'GaU':
                    continue
                if Mname == 'GT' and polish != 'GaU':
                    continue
            i = i + 1
            slice = at_samples(polish)
            if folder != 'all':
                data = M.res[polish][err][0]
                if len(data) ==0:
                    continue
                y = np.median(data,axis=0)
                mm[M.name,polish] += [y]
                ylabel = 'Median ' + err_name[err]
                hist, bine = np.histogram(data[:,slice], Nbins, range=(0,xrange), density=False) # method an polish
                mh[M.name, polish] += [data[:, slice]]
            else:
                data = np.stack(mm[M.name, polish]) # mean meadian
                y = data.mean(axis=0)
                data = np.hstack(mh[M.name, polish])  # mean meadian
                hist, bine = np.histogram(data, Nbins, range=(0,xrange), density=False) # method an polish
                ylabel = 'Mean Median ' + err_name[err]
            # y = np.array([np.median(v, axis=0) for v in getattr(M, err)]).mean(axis=0)
            x = np.arange(1,len(y)+1)*100
            name = M.name
            # print(name)
            color = cc[i]
            if M.name == 'Oracle':
                color = 'black'
            if M.name == 'GT':
                color = 'gray'                
            if polish != 0:
                x = (np.array([0] + [*np.arange(1, len(y))*5-1])+1)*100
                style = '--' + markers[i]
                if polish != 'GaU':
                    color = None
                    # style += cc[i]
                pname = f'+LMA({polish})'
                pname = pname.replace('TRUNCATED_LE_ZACH', 'Lee-Zach')
                pname = pname.replace('TRUNCATED', 'Truncated')
                name += pname
                alpha = 0.5
            else:
                style = '-' + markers[i]
                alpha= 1
                if hasattr(M, 'best_hyperparam'):
                    # name += (r' ($\tau{=}'+f'{M.best_hyperparam:2.2f}$)')
                    pass
                name = name.replace('gamma=30.0', '').replace('gamma=10.0', '').replace('    ', ' ')
            plt.figure(f1)
            mevery = 12 + i
            if polish != 0:
                mevery = i-4
            plt.plot(x, y, style, color = color, label=name, alpha = alpha, markevery=[mevery])
            plt.figure(f2)
            if M.name in ['Oracle', 'GT', 'GaU', 'RANSAC'] and polish in [0,'GaU']:
                plt.plot(bine[:-1], hist, style, color=color, label=name, alpha=alpha, markevery=[20+3*i])
    plt.figure(f1)
    # if dataset_info != PhotoTourismRootSIFT:
    plt.legend(loc=1, fontsize=10)
    plt.grid()
    plt.xlim(100)
    plt.ylim(bottom=0)
    # plt.axvline(1000,color='k',linestyle='--')
    plt.ylabel(ylabel)
    plt.xlabel('Minimal samples inspected')
    plt.title(dataset_info.name + ' ' + folder.replace('all','all test'))
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # methods = load_running(polish=0)
    # M = methods[0]
    # y = np.array([np.median(v, axis=0) for v in M.running_s]).mean(axis=0)
    # x = np.arange(1, len(y)+1)*100
    # plt.plot(x, y, ':' + cc[0], label=None)
    # plt.ylabel('Meadian Score')
    
    if folder  == 'st_pauls_cathedral':
        plt.ylim(top =5)
        # plt.yticks(np.arange(0,5.1,0.5))
    if folder in ['london_bridge', 'united_states_capitol']:
        plt.ylim(top=25)
        # plt.yticks(np.arange(0, 25.1, 5))
        # plt.grid(visible=True, which='minor', axis='y')
    if folder == 'all' and dataset_info == PhotoTourismRootSIFT or dataset_info == PhotoTourismSPSG:
        plt.ylim(top=10)
        # plt.yticks(np.arange(0, 10.1, 1))
        # plt.grid(visible=True, which='minor', axis='y')
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    
    plt.xlim(100,4000)
    plt.xticks([100, 500, 1000, 2000, 3000, 4000])
    # plt.xscale('log')
    # plt.yscale('log')
    outf = res_root + f'{err}-{folder}.pdf'
    print(outf)
    force_path(outf)
    plt.draw()
    plt.savefig(outf, bbox_inches='tight', pad_inches=0.0)
    # if folder in ['florence_cathedral_side', 'st_pauls_cathedral', 'sagrada_familia', 'united_states_capitol', 'all']:
    # if folder in ['st_pauls_cathedral', 'london_bridge', 'united_states_capitol', 'all']:
        # plt.show(block = True)
    # plt.show()

    plt.figure(f2)
    plt.legend(loc=1)
    plt.grid()
    plt.ylabel(f'Number of pairs with error ±{xrange/ Nbins/2}')
    plt.xlabel('Error value ' + err_name[err])
    plt.gcf().tight_layout()
    plt.draw()
    plt.savefig(res_root + f'distribuiotn_{err}-{folder}.pdf', bbox_inches="tight", pad_inches=0.0, format='pdf', transparent=True)
    if folder in ['st_pauls_cathedral', 'london_bridge', 'united_states_capitol', 'all']:
        plt.show()
    plt.close(f1)
    plt.close(f2)
    
    # break

# %% Error distribution starting LO with GT

#polishes = [0, 'GaU', 'TRUNCATED', 'TRUNCATED_LE_ZACH']
err = 'running_r'
Nbins = 50
plt.figure()
for i, polish in enumerate(polishes):
    if polish == 0:
        continue
    Mname = 'GT'
    data = []
    for folder in dataset_info.test:
        methods = load_running(folder + '/', 'all')
        name_to_method = {m.name: m for m in methods}
        M = name_to_method[Mname]
        data += [M.res[polish][err][0][:,0]]
    data = np.hstack(data)
    hist, bine = np.histogram(data, Nbins, range=(0, xrange), density=False)  # method an polish
    br = scipy.stats.bootstrap((data,), np.median, confidence_level=0.95, method='BCa', n_resamples=1000)
    ci = br.confidence_interval
    v = np.median(data)
    d = (ci.high - ci.low)/2
    formatted = format_std(v, d)
    name = M.name
            # print(name)
    color = cc[i] 
    pname = f'+LMA({polish})'
    pname = pname.replace('TRUNCATED_LE_ZACH', 'Lee-Zach')
    pname = pname.replace('TRUNCATED', 'Truncated')
    name += pname
    print(f'\t {name}: {formatted}', end='')
    plt.plot(bine[:-1], hist, '-', marker = markers[i], color=color, label=name, alpha=alpha, markevery=[3*i])
plt.legend(loc=1)
plt.grid()
plt.ylabel(f'Number of pairs with error ±{xrange/ Nbins/2}')
plt.xlabel(err_name[err])
plt.gcf().tight_layout()
plt.draw()
plt.savefig(res_root + f'distribuiotn_GT_{err}-all.pdf', bbox_inches="tight", pad_inches=0.0, format='pdf', transparent=True)
plt.show()
plt.close(plt.gcf())

# plt.figure()

# %%

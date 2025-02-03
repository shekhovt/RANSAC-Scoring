# %%
import os
import sys
if __name__ == "__main__":
    __name__ = 'score_learn.res_table.py'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
else:
    __run__ = False

import pandas as pd
from .load_data import *

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Important parameters
# dataset_info = PhotoTourismSPSG
# dataset_info = PhotoTourismRootSIFT
dataset_info = KITTI
# dataset_info = ETH3D
atsamples = 4000
polishes = [0, 'GaU', 'TRUNCATED', 'TRUNCATED_LE_ZACH']
# all_methpds = ['RANSAC', 'MSAC', 'MAGSAC++','GaU','Learned','Oracle','GC-MAGSAC++']
# selected = ['GaU', 'Oracle']  # 'RANSAC', 'MSAC',
all_methpds = ['GaU', 'MSAC', 'MAGSAC++', 'RANSAC', 'Learned', 'Oracle']
selected = ['RANSAC', 'MSAC', 'MAGSAC++', 'GaU' , 'Learned', 'Oracle']

# polishes = [0]

# Eval_GCMAGSAC = False

# dataset_info = PhotoTourismRootSIFT
res_root = f'results/{dataset_info.name}/'
results_path = res_root + f'polish=all/'

# if Eval_GCMAGSAC:
#     res_root += 'GCMAGSAC/'

out1 = res_root + f'results-0.txt'
out2 = res_root + f'results-polish.txt'

lists  = []

def postprocess(res_list):
    for d in res_list:
        if 'ML' in d['method']:
            d['method'] = 'Learned'
        d['err_key'] = d['err_key'].replace('best_', '').replace('running_', '')
        val = d['val']
        try:
            v, s = val.split('±')
            v = float(v)
            s = float(s)
        except:
            v = float(val)
            s = float('NaN')
        if d['stat'] == 'median':
            d['val'] = f'{v:.3f}'
        else:
            d['val'] = f'{v:.3f}'
        d['ci'] = s
        d['err_key'] = '$' + d['err_key'] + '$'

test_set = dataset_info.test
table_file = results_path + f'testr-{atsamples}-table.pkl'
res_list = load_object(table_file)
postprocess(res_list)
lists = res_list  # concatenate all records
        

# mean_s = []
# for polish in polishes:
#     results_path = res_root + f'polish={polish}/'
#     table_file = results_path + f'test_table.pkl'
#     try:
#         res_list = load_object(table_file)
#     except:
#         print(f'Missing {table_file}')
#         continue
#     lists += res_list # concatenate all records


df = pd.DataFrame.from_records(lists, index='method', coerce_float=False, nrows=None)

# %%
df1 = df.drop(['joint'], axis=1)
# dfs = dict(tuple(df1.groupby('stat')))
SS = ''
# for k in dfs.keys():
    # print(f"stat={k}")
# d = dfs[k]
d = df1.reset_index()
d['polish'] = pd.Categorical(d['polish'], polishes)
d['err_key'] = pd.Categorical(d['err_key'], ['$e$','$r$','$t$'])
d['method'] = pd.Categorical(d['method'], selected)
# ci = d['ci']
# d = d.drop(['ci'], axis=1)
# df2 = d.drop(['stat'], axis=1)
df2 = d
df3 = df2.set_index(['method', 'err_key'])
dg = df3.groupby(['polish', 'stat', 'method', 'err_key'])
df4 = dg['val'].aggregate('first').unstack([0,1])
dfg_ci = df3.groupby(['stat', 'method', 'err_key'])['ci'].aggregate('mean').unstack(0)
# dfg_ci = dfg_ci.groupby(['stat', 'method', 'err_key'])
# dfg_ci = dfg_ci.set_index(['method', 'err_key'])
print(dfg_ci)
ci0 = dfg_ci['AUC_10'].apply(lambda x: f'±{x:.3f}')
ci1 = dfg_ci['median'].apply(lambda x: f'±{x:.2f}')
df4['Conf AUC'] = ci0
df4['Conf Med'] = ci1
display(df4)
ss = df4.style.to_latex()
ss = ss.replace('±', '$\pm$')
ss = ss.replace('err_key','err')
ss = ss.replace('$r$', '$e_R$')
ss = ss.replace('$t$', '$e_t$')
ss = ss.replace('AUC_10',r'AUC')
ss = ss.replace('median', 'Med')
ss = ss.replace(r'\multirow[c]{3}', r'\midrule'+ '\n' + r'\multirow{3}')
SS += ss
# ss = df.to_latex(index=True)
# ss = ss.replace('±', '$\pm$')
f = open(out2, "w")
f.write(SS)
f.close()

# # %%
# df.reset_index().groupby(['id', 'polish'])['val'].aggregate('first').unstack()

# %%


# selected = ['RANSAC', 'MSAC', 'GaU', 'Oracle']
# polishes = [0]
# polishes = [0, 'GaU', 'TRUNCATED', 'TRUNCATED_LE_ZACH']

def postprocess2(res_list):
    for d in res_list:
        if 'ML' in d['method']:
            d['method'] = 'Learned'
        d['err_key'] = d['err_key'].replace('best_', '').replace('running_', '')
        d['err_key'] = '$' + d['err_key'] + '$'
        val = d['val']
        try:
            v, s = val.split('±')
            v = float(v)
            s = float(s)
        except:
            v = float(val)
            s = float('NaN')
        if d['stat'] == 'median':
            d['val'] = f'{v:.2f}±{s:.3f}'
        else:
            d['val'] = f'{v:.3f}±{s:.3f}'
        # d['val'] = f'{v:.2f}±{s:.2f}'
        # d['polish'] = 'LO('+str(d['polish']) + ')'

res_list = load_object(table_file)
postprocess2(res_list)
lists = res_list  # concatenate all records

df = pd.DataFrame.from_records(lists, index='method', coerce_float=False, nrows=None)

df1 = df.drop(['joint'], axis=1)
# dfs = dict(tuple(df1.groupby('stat')))
# for k in dfs.keys():
# print(f"stat={k}")
# d = dfs[k]
d = df1.reset_index()
# d['polish'] = pd.Categorical(d['polish'], [0])
d = d[d['polish'] == 0]
d = d.drop(['polish'], axis=1)
d['err_key'] = pd.Categorical(d['err_key'], ['$e$', '$r$', '$t$'])
d['method'] = pd.Categorical(d['method'], selected)
df2 = d
df3 = df2.set_index(['method'])
df4 = df3.groupby(['stat', 'err_key', 'method'])
df4 = df4['val'].aggregate('first').unstack([0,1])
display(df4)

# %%
d = df1.reset_index()
d['polish'] = pd.Categorical(d['polish'], polishes[1:])
# d = d.drop(['polish'], axis=1)
d['err_key'] = pd.Categorical(d['err_key'], ['$e$', '$r$', '$t$'])
# d['method'] = pd.Categorical(d['method'], ['GaU'])
df2 = d[d['method'] == 'GaU']
# df2 = df2.drop(['method'], axis=1)
df3 = df2.set_index(['polish'])
df5 = df3.groupby(['stat', 'err_key', 'polish'])
df5 = df5['val'].aggregate('first').unstack([0, 1])
display(df5)

# %%
# dg = df3.groupby(['polish'])
SS = ''
for dd in [df4, df5]:
    ss = dd.style.to_latex()
    ss = ss.replace('±', '$\pm$')
    ss = ss.replace('err_key', 'err')
    ss = ss.replace('$r$', '$e_R$')
    ss = ss.replace('$t$', '$e_t$')
    ss = ss.replace('AUC_10', r'mAA')
    ss = ss.replace('median', 'Med')
    ss = ss.replace(r'\multirow[c]{3}', r'\midrule' + '\n' + r'\multirow{3}')
    ss = ss.replace(r'\multirow[c]{3}', r'\midrule' + '\n' + r'\multirow{3}')
    for p in polishes:
        pname = 'LO('+str(p)+')'
        ss = ss.replace(r'\multirow[c]{4}{*}{'+pname+'}', '\\midrule\n' r'\multicolumn{7}{c}{' + pname + r'}\\' + '\n\\midrule\n')
    ss = ss.replace('TRUNCATED_LE_ZACH', 'PoseLib (Le-Zach)')
    ss = ss.replace('TRUNCATED', 'PoseLib (Truncated)')
    if dd is df5:
        ss = ss.replace('GaU', 'IRLS-LMA GaU')
    # ss = ss.replace('LO(0)', 'Minimal Samples')
    ss = ss.replace('polish', ' LO')
    # ss = ss.replace('\n &', '\n')
    SS += ss
# ss = df.to_latex(index=True)
# ss = ss.replace('±', '$\pm$')
print(SS)
f = open(out1, "w")
f.write(SS)
f.close()

# %%

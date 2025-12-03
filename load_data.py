#%%
import os
import sys
if __name__ == "__main__":
    __name__ = 'score_learn.load_data.py'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
else:
    __run__ = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

import torch.utils.data as data
import copy
import h5py
import numpy as np
import yaml

from  .tools import *
from  .score_weights import *
from  .score_weights import SampsonM
from .model_H import normalize_points

PhotoTourismSPSG = dotdict(
name = 'PhotoTourismSPSG',
Fundamental = False,
root='/home/weitong/code/differentiable_ransac_private/saved_residuals/',
RT_path_train='/mnt/personal/weitong/RANSAC-Tutorial-Data/train/',
RT_path_test='/mnt/personal/weitong/raw_data/RANSAC-Tutorial-Data-Test/',
test = [
    'florence_cathedral_side',
    'british_museum',
    'lincoln_memorial_statue',
    'london_bridge',
    'milan_cathedral',
    'mount_rushmore',
    'piazza_san_marco',
    'reichstag',
    'sagrada_familia',
    'st_pauls_cathedral',
    'united_states_capitol'
    ]
,
val = [
    'buckingham_palace', # 0
    'brandenburg_gate', # 1
    'colosseum_exterior', # 2
    'grand_place_brussels', # 3 
    'notre_dame_front_facade', # 4
    'palace_of_westminster', # 5
    'pantheon_exterior', # 6
    'prague_old_town_square', # 7
    # 'sacre_coeur',
    'taj_mahal', # 8
    'trevi_fountain', # 9
    'westminster_abbey' # 10
    
],
train = ['st_peters_square']
)

PhotoTourismRootSIFT = copy.deepcopy(PhotoTourismSPSG)
PhotoTourismRootSIFT.root = '/mnt/personal/weitong/saved_residuals_tutorial_0.9/'
# PhotoTourismRootSIFT.val_path = '/mnt/personal/weitong/saved_models_tutorial_0.9_4950/'
PhotoTourismRootSIFT.val_path = '/mnt/personal/weitong/saved_residuals_tutorial_0.9_4950/'
PhotoTourismRootSIFT.name = 'PhotoTourismRootSIFT'

ScanNet = dotdict(
    name='ScanNet',
    Fundamental=False,
    root='/home/weitong/code/differentiable_ransac_private/saved_residuals/scannet/',
    test=['test'],
    val=['train'],
    train=['train'],
)

LAMAR = dotdict(
    name='LAMAR',
    Fundamental=False,
    root='/home/weitong/code/differentiable_ransac_private/saved_residuals/',
    RT_path = '/mnt/personal/weitong/features/sp_sg_features/',
    test=['cab_test', 'lin_test', 'hge_test'],
    val=['cab_train', 'lin_train', 'hge_train'],
    train=['cab_train'],
)

ETH3D = dotdict(
    name='ETH3D',
    Fundamental=False,
    root='/mnt/personal/weitong/eth3d_test_models/',
    RT_path = '/mnt/personal/weitong/features/hsac/SuperGluePretrainedNetwork/',
    test=['eth3d'],
    val=['eth3d_train'],
    train=['eth3d_train'],
)

KITTI = dotdict(
    name='KITTI',
    Fundamental=True,
    root='/home/shekhole/data/KITTI/', #'/home/weitong/code/differentiable_ransac_private/saved_residuals/KITTI/',
    RT_path_train = '/mnt/personal/weitong/RootSIFT_features/KITTI/train_data_rs/', # where do we get KITTI R,t ground truth?
    RT_path_test = '/mnt/personal/weitong/RootSIFT_features/KITTI/test_data_rs/',
    test=['test'],
    val=['train'],
    train=['train'],
)

ScanNet = dotdict(
    name='ScanNet',
    Fundamental=False,
    root='/home/weitong/code/differentiable_ransac_private/saved_residuals/scannet/',
    test=['test'],
    val=['train'],
    train=['train'],
)

HEB = dotdict(
    name='HEB',
    type='H',
    root='./data/HEBHomographyDataset/',
    RT_path_train='./data/HEBHomographyDataset/training_and_validation',
    RT_path_test='./data/HEBHomographyDataset/test/',
test = [
    'Ellis_Island',
    'Piazza_del_Popolo',
    'Tower_of_London',
    'Vienna_Cathedral',
    'Madrid_Metropolis',
    'Roman_Forum',
    'Union_Square',
    'Yorkminster'
    ]
,
val = [
    'Alamo',
    'NYC_Library'
]    
)

datasets = [PhotoTourismSPSG, PhotoTourismRootSIFT, ScanNet, LAMAR, ETH3D, KITTI, HEB]


def username():
    try: 
        login = os.getlogin()
        if login == 'shekhovt' or login == 'shekhole':
            return 'shekhole'
        else:
            return login
    except:
        return 'weitong'

def mirror(source):
    if username() == 'shekhole':
        compressed_data_root = '/tmp/RANSAC/'
        target = source.replace('/home/weitong/code/',compressed_data_root)
        target = target.replace('/mnt/personal/weitong/', compressed_data_root)
        target = target.replace('/home/shekhole/data/', compressed_data_root)
        if not os.path.exists(target):
            print("We dont't have " + target)
            force_path(target)
            # mirror
            os.system(f"rsync -r -v {username()}@login3.rci.cvut.cz:{source} {target}")
        return target
    else:
        return source


class ResidualData(data.Dataset):
    
    def loat_Rt(self, dataset_info, folder):
        if dataset_info == PhotoTourismRootSIFT or dataset_info == PhotoTourismSPSG:
            if folder in dataset_info.test:
                pth = dataset_info.RT_path_test
            else:
                pth = dataset_info.RT_path_train
            mirror(pth + f"{folder} /")
            fr = h5py.File(mirror(pth + f"{folder}/R.h5"), 'r')
            ft = h5py.File(mirror(pth + f"{folder}/T.h5"), 'r')
            # fm = h5py.File(mirror(pth + f"{folder}/matches.h5"), 'r')
            # fsnn = h5py.File(mirror(pth + f"{folder}/match_conf.h5"), 'r')
            for i, f in enumerate(self.Data.data):
                names = f['files'].split('/')[-1].split('_')
                names_23 = names[2] + '_' + names[3]
                names_01 = names[0] + '_' + names[1]
                self.Data.data[i]['gt_R'] = fr[names_23].__array__() @(fr[names_01].__array__().T)
                self.Data.data[i]['gt_t'] = ft[names_23].__array__() - self.Data.data[i]['gt_R']@(ft[names_01].__array__())
                pair = names_01 + '-' + names_23
                # snn = fsnn[pair].__array__()
                # m = fm[pair].__array__()

        elif dataset_info == LAMAR or dataset_info == ScanNet:
            pth = dataset_info.RT_path
            mirror(pth)
            src_gt = f"{pth}{folder.split('_')[0]}_sp_sg_outdoor_noresize/"
            for i, f in enumerate(self.Data.data):
                names = f['files'].split('/')[-1].split('_')
                gt_file = np.load(mirror(src_gt + names[0] + '_' + names[1] + '_matches.npz'))
                self.Data.data[i]['gt_R'] = gt_file['R_1_2']
                self.Data.data[i]['gt_t'] = gt_file['T_1_2']
        
        elif dataset_info == ETH3D:
            pth = dataset_info.RT_path
            mirror(pth)
            src_gt = f"{pth}{folder}_high_sp_sg_outdoor/"
            for i, f in enumerate(self.Data.data):
                names = f['files'].split('/')[-1].split('_')
                gt_file = np.load(mirror(src_gt + names[0] + '_' + names[1] + '_' + names[2] + '_' + names[3] + '_matches.npz'))
                self.Data.data[i]['gt_R'] = gt_file['R_1_2']
                self.Data.data[i]['gt_t'] = gt_file['T_1_2']
                # gt_file = np.load(mirror(src_gt + names[0] + '_' + names[1] + '_matches.npz'))
                # self.Data.data[i]['gt_R'] = gt_file['R_1_2']
                # self.Data.data[i]['gt_t'] = gt_file['T_1_2']
        
        elif dataset_info == KITTI:
            if folder in dataset_info.test:
                pth = dataset_info.RT_path_test
            elif folder in [dataset_info.val, dataset_info.train]:
                pth = dataset_info.RT_path_train
            else:
                raise RuntimeError(f'{folder} not in test/train/val')
            mirror(pth)
            for i, f in enumerate(self.Data.data):
                names = f['files'].split('/')[-1].split('_')
                gt_file = np.load(mirror(f'{pth}/pair_{names[0]}_{names[1]}.npy'), allow_pickle=True)
                self.Data.data[i]['gt_R'] = gt_file[7]
                self.Data.data[i]['gt_t'] = gt_file[8]

    def load_magsac_selected_model(self, dataset_info, folder):

        "fetch the models selected by MAGSAC++ scoing on models built by PROSAC, no polish"
        models = h5py.File(mirror(f"/home/weitong/code/score/ransac/magsac_selected_no_polish_{folder}.h5"), 'r')
        for i, f in enumerate(self.Data.data):
            names = f['files'].split('/')[-1].split('_')
            GCdata = models[names[0] + '_' + names[1] + '_' + names[2] + '_' + names[3]]
            C = np.array(GCdata['correspondences'])[0]
            C0 = self.Data.data[i]['correspondences']
            assert(C.shape == C0.shape)
            assert (np.abs(C0-C).max() < 1e-5)
            self.Data.data[i]['magsac_selected'] = GCdata['model'].__array__()

    def __init__(self, dataset_info, folder, padding=False, size=3000) -> None:
        if folder in dataset_info.val and dataset_info.val_path is not None:
            source_data_root = dataset_info.val_path
        else:
            source_data_root = dataset_info.root
        source_path = source_data_root + f"{folder}/"
        target_path = mirror(source_path)
        self.files = [target_path + f for f in os.listdir(target_path) if f.endswith(".npz")]
        self.padding = padding
        self.size = size
        self.sqrt = True # depricated
        self.Data = None
        self.src = target_path
        self.F = dataset_info.Fundamental
        file = self.src + 'compressed.pkl'

        if os.path.exists(file):
            print('Loaded compressed data')
            self.Data = load_object(file)
        else:
            print('Compressing')
            self.Data = self.compress()
            self.loat_Rt(dataset_info,folder)            
            file = self.src + 'compressed.pkl'
            save_object(file, self.Data)
            print('compressed size: ' + str(os.path.getsize(file) / 1024/1024) + ' Mb')

            if username() == 'shekhole':
                for f in self.files:
                    os.remove(f)
        # update old data with Rt if needed
        # self.loat_Rt(dataset_info, folder)
        if 'gt_R' not in self.Data.data[0]:
            self.loat_Rt(dataset_info, folder)
            file = self.src + 'compressed.pkl'
            save_object(file, self.Data)
            print('Recompressed size: ' + str(os.path.getsize(file) / 1024/1024) + ' Mb')
        # load the MAGSAC++ scoring seleted models
        if dataset_info == PhotoTourismRootSIFT and folder in dataset_info.test:
            self.load_magsac_selected_model(dataset_info, folder)

    def compress(self):
        padding = self.padding
        self.padding = False
        Data = SimpleNamespace()
        Data.meta = SimpleNamespace()
        Data.data = []
        cmax = 0
        for i in range(len(self)):
            data = self[i]
            del data['residuals']
            Data.data.append(data)
            cmax = np.maximum(np.nan_to_num(cmax), data['correspondences'].shape[0]) 
            
        print(f'cmax: {cmax}')
        Data.meta.cmax = cmax
        Data.meta.len = len(Data.data)
        self.padding = padding
        return Data

    def __len__(self):
        if self.Data is not None:
            return self.Data.meta.len
        else:
            return len(self.files)
    
    # def compute_residuals(self,data):
    #     correspondences = data['correspondences']
    #     C = torch.tensor(correspondences).float().cuda()  # [N 4]
    #     x = torch.cat([C[:, :2], C.new_ones((C.shape[0], 1))], dim=-1)
    #     y = torch.cat([C[:, 2:], C.new_ones((C.shape[0], 1))], dim=-1)
    #     K1 = torch.tensor(data['K1'], dtype=torch.float) # [3, 3] -- paired with x
    #     K2 = torch.tensor(data['K2'], dtype=torch.float) # [3, 3] -- paired with y        
    #     models = np.float32(data['models'])
    #     # converting to fundamental matrix and recomputing residuals anew
    #     K1I = K1.inverse().cuda()
    #     K1 = K1.cuda()
    #     K2I = K2.inverse().cuda()
    #     K2 = K2.cuda()
    #     E = torch.tensor(models).to(device='cuda', dtype=torch.float32)  # [M, 3, 3]
    #     # r = SampsonM(y, x, F) # y'F x
    #     # scale = (K1[0,0]+ K1[1,1] + K2[0,0] + K2[1,1])/4
    #     # print(K1[0,2].item())
    #     # unnormalize
    #     x = x @ K1.T  # (K1)x in the format [n,3]
    #     y = y @ K2.T  # (K2)y in the format [n,3]
    #     # transform E to F
    #     F = torch.einsum('ij, mik, kl -> mjl', K2I,E, K1I)  # K2^{-T} F K1^{-1}
    #     # r_new, logJJ = SampsonJJ(y, x, F) # y'F x
    #     r_new = SampsonM(y, x, F)  # y'F x
    #     # logJJ = r_new.new_zeros(r_new.shape[0]) # [M]
    #     models = F.cpu().numpy()
    #     residuals = r_new.abs().cpu().numpy()
    #     data['residuals'] = residuals

    def padd_data(self, data):
        N = self.Data.meta.cmax
        C = data['correspondences']
        C = np.concatenate((C, np.full((max(0, N - C.shape[0]), C.shape[1]), np.float32(np.inf))), axis=0)
        data['correspondences'] = C        
        
        if 'residuals' in data:
            R = data['residuals']
            R = np.concatenate((R, np.full((R.shape[0], max(0, N - R.shape[1])), np.float32(np.inf))), axis=-1)
            data['residuals'] = R            
    
    def __getitem__(self, index):
        if self.Data is not None:
            data = copy.deepcopy(self.Data.data[index])
            if self.F:
                # normalize correspondences
                C = torch.tensor(data['correspondences'])
                x = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
                y = torch.cat([C[..., 2:], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
                K1 = torch.tensor(data['K1']) # [3, 3] -- paired with x
                K2 = torch.tensor(data['K2']) # [3, 3] -- paired with y
                K1I = K1.inverse()
                K2I = K2.inverse()
                x = torch.einsum('ij, nj -> ni', K1I, x)
                y = torch.einsum('ij, nj -> ni', K2I, y)
                x = x[:,0:2]/x[:,-1:]
                y = y[:, 0:2]/y[:, -1:]
                C = torch.cat([x,y], dim=-1)
                data['correspondences'] = C.numpy()
                # convert F to E:
                F = torch.tensor(data['models'])
                E = torch.einsum('ij, mik, kl -> mjl', K2, F, K1)  # K2^{-T} F K1^{-1}
                data['models'] = E
            # if not 'residuals' in data:
            # self.compute_residuals(data)
            self.padd_data(data)
            # data["is_F"] = self.F # fundamental matrix flag
            data["is_F"] = False  # fundamental matrix flag
            return data
        else:
            return self.__getitem__load(index)
        
    def __getitem__load(self, index):
        data = np.load(self.files[index])
        correspondences = np.float32(data['correspondences'])
        # K1 = torch.tensor(data['K1'], dtype=torch.float) # [3, 3] -- paired with x
        # K2 = torch.tensor(data['K2'], dtype=torch.float) # [3, 3] -- paired with y        
        models = np.float32(data['models'])
        errors = np.float32(data['errors'])
        # print(scale)
        # #
        # if False: # converting to fundamental matrix and recomputing residuals anew
        #     # C = torch.tensor(correspondences).float().cuda()  # [N 4]
        #     # x = torch.cat([C[:, :2], C.new_ones((C.shape[0], 1))], dim=-1)
        #     # y = torch.cat([C[:, 2:], C.new_ones((C.shape[0], 1))], dim=-1)
        #     K1I = K1.inverse().cuda(); K1 = K1.cuda()            
        #     K2I = K2.inverse().cuda(); K2 = K2.cuda()            
        #     E = torch.tensor(models).to(device='cuda',dtype=torch.float32) # [M, 3, 3]
        #     # r = SampsonM(y, x, F) # y'F x
        #     # scale = (K1[0,0]+ K1[1,1] + K2[0,0] + K2[1,1])/4
        #     # print(K1[0,2].item())
        #     # unnormalize
        #     x = x @ K1.T # (K1)x in the format [n,3]
        #     y = y @ K2.T # (K2)y in the format [n,3]
        #     # transform E to F
        #     F = torch.einsum('ij, mik, kl -> mjl',K2I,E,K1I) # K2^{-T} F K1^{-1}
        #     # r_new, logJJ = SampsonJJ(y, x, F) # y'F x
        #     r_new = SampsonM(y, x, F)  # y'F x
        #     # logJJ = r_new.new_zeros(r_new.shape[0]) # [M]
        #     models = F.cpu().numpy()
        #     residuals = r_new.abs().cpu().numpy()
        #     # logJJ = logJJ[idx]            
        # else:
        #     r_old = np.float32(data['residuals'])
        #     scale = (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]).item()/4 # this is scale for plain residuals
        #     if r_old.min() < 0 or (r_old != r_old).any():
        #         print(r_old.min())
        #         print(self.files[index])
        #     assert self.sqrt
        #     if self.sqrt:
        #         # assert (r_old >=0).all()
        #         r_old = np.sqrt(r_old)
        #         residuals = r_old * scale
        #     else:
        #         residuals = r_old * scale**2
        #     # logJJ = residuals.new_zeros(residuals.shape[0])  # [M]
        # # print((K1[0,0]/K1[0,2]).item())
        # #
        # if residuals.shape[0] < 1001:
        #     print(self.files[index] + ' models:' + str(residuals.shape[0]))
        idx = np.arange(1001)
        # idx = np.random.permutation(residuals.shape[0]-1)[0:1001] # randomly subsample 1001 from all but last model
        for k in range(10):
            idx[idx>=models.shape[0]] = idx[idx>=models.shape[0]] - models.shape[0] # just loop over existing moldes and copy
        idx[-1] = models.shape[0]-1 # fetch GT model as last
        # residuals = residuals[idx,:]
        errors = errors[idx]
        models = models[idx]
        # correspondences = correspondences[idx] # different length, need to apply padding
        # if correspondences.shape[0]>  self.size: import pdb;pdb.set_trace()
        num_pts = correspondences.shape[0]
        if self.padding:
            # residuals = np.concatenate((residuals, np.full((residuals.shape[0], max(1, self.size - residuals.shape[1])), np.float32(np.inf))), axis=-1)
            correspondences = np.concatenate((correspondences, np.full((correspondences.shape[0], max(1, self.size - correspondences.shape[1])), np.float32(np.inf))), axis=-1)
        r = {
            'num_pts': num_pts,
            'errors': errors,
            'residuals': None,
            'files': self.files[index],
            'K1': data['K1'],
            'K2': data['K2'],
            'models': models,
            'correspondences': correspondences
            # 'logJJ': logJJ, # padd with zeros
            # 'correspondences': correspondences,
        }
        return r



def test_loader(loader):
    # sqrt = True  # SPSG features
    # residual_dataset = ResidualData(datasets[1], folder = 'british_museum', padding=True) # padding the residuals with infs
    # data_loader = torch.utils.data.DataLoader(
    #         residual_dataset,
    #         batch_size=32,
    #         num_workers=0,
    #         pin_memory=True,
    #         shuffle=True,
    #     )

    # Cx = []
    for idx, data in enumerate(loader):
        print(data.keys())
        break
        # residuals = data['residuals']
        # gt_errors = data['errors']
        Cx += [data['K1'][:,0,2]]
        if idx>10:
            break
        # import pdb; pdb.set_trace()
    # C = np.concatenate(Cx)
    # C = np.unique(C)
    # print(C)


def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key in keys:
            dict_to_load[key] = f[key][()]
    return dict_to_load

class H_dataset(data.Dataset):
    def __init__(self, dataset_info, scene, padding=False, size=3000, snn_threshold = 0.9) -> None:
        self.padding = padding
        self.size = size
        config_path = dataset_info.root + '/dataset_configuration.yaml'
        # Loading the configuration file
        with open(config_path, "r") as stream:
            try:
                configuration = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        
        scene_f = None
        for s in configuration['TEST_SCENES'] + configuration['TRAIN_SCENES']:
            if scene == s['name']:
                scene_scale = s['scale']
                scene_f = s['filename']
                if s in configuration['TEST_SCENES'] and not s in configuration['TRAIN_SCENES']:
                    root = dataset_info.RT_path_test
                else:
                    root = dataset_info.RT_path_train

        if scene_f is None:
            raise RuntimeError()

        source_path = os.path.join(root, scene_f)
        target_path = mirror(source_path)
        self.padding = padding
        self.size = size # IDK
        self.Data = None
        self.src = target_path
        self.kind = 'homography'
        self.scene_scale = scene_scale
        self.scene = scene
        self.snn_threshold = snn_threshold
        # select a subset of image pairs?
        file = self.src + '_compressed.pkl'

        if os.path.exists(file):
            print('Loaded compressed data')
            self.Data = load_object(file)
            self.files = self.Data.meta.files
        else:
            print('Compressing')
            self.Data = self.compress()
            save_object(file, self.Data)
            print('compressed size: ' + str(os.path.getsize(file) / 1024/1024) + ' Mb')

    def compress(self):
        padding = self.padding
        self.padding = False
        Data = SimpleNamespace()
        Data.meta = SimpleNamespace()
        Data.data = []
        cmax = 0
        # fetch "files" -- number of pairs in the dataset
        print(f"Loading scene '{self.scene}'")
        print(f"The scene scale is {self.scene_scale}")
        self.data = load_h5(self.src)
        self.files = sorted([x.replace('corr_','') for x in self.data.keys() if x.startswith('corr_')])
        Data.meta.files = self.files
        Data.meta.scale = self.scene_scale
        Data.meta.snn_threshold = self.snn_threshold
        print(f"Number of pairs: {len(self.files)}")
        #
        for i in range(len(self)):
            data = self[i]
            Data.data.append(data)
            cmax = np.maximum(np.nan_to_num(cmax), data['correspondences'].shape[0]) 
        #  
        print(f'cmax: {cmax}')
        Data.meta.cmax = cmax
        Data.meta.len = len(Data.data)
        self.padding = padding
        return Data

    def __len__(self):
        if self.Data is not None:
            return self.Data.meta.len
        else:
            return len(self.files)
        
    def padd_data(self, data):
        N = self.Data.meta.cmax
        C = data['correspondences']
        C = np.concatenate((C, np.full((max(0, N - C.shape[0]), C.shape[1]), np.float32(np.inf))), axis=0)
        data['correspondences'] = C

    def __getitem__load(self, index):
        p = self.files[index]
        data = self.data
        corr = data[f'corr_{p}'] # The SIFT unnormalized correspondences [num_pts x 9]: [X1, Y1, X2, Y2, angle1, angle2, scale1, scale2, snn_ratio, is_inlier_gt]
        snn_ratio = corr[:,-2]
        mask = snn_ratio < self.snn_threshold
        C = corr[mask, 0:4].astype(np.float32)
        pose = data[f'pose_{p}'].astype(np.float64) # The ground truth relative pose coming from the COLMAP reconstruction, 3x4 matrix ?
        size1 = data[f"size_{ '_'.join(p.split('_')[0:3]) }"].astype(np.float32) # The size of the source image
        size2 = data[f"size_{ '_'.join(p.split('_')[3:6]) }"].astype(np.float32) # The size of the destination image
        K1 = data[f"K_{ '_'.join(p.split('_')[0:3]) }"].astype(np.float32) # The intrinsic matrix of the source image 3x3, single focal length, principle point at size/2
        K2 = data[f"K_{ '_'.join(p.split('_')[3:6]) }"].astype(np.float32)  
        num_pts = C.shape[0]
        r = {
            'correspondences': C,
            'K1': K1,
            'K2': K2,
            'size1': size1,
            'size2': size2,
            'num_pts': num_pts,
            'files': p,
            'gt_R': pose[:,0:3],
            'gt_t': pose[:,3],
            'models': torch.empty(size=(0,3,3)).numpy(),
        }
        return r

    def __getitem__(self, index):
        if self.Data is not None:
            data = copy.deepcopy(self.Data.data[index])
            # normalize correspondences
            C = torch.tensor(data['correspondences'])
            x = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
            y = torch.cat([C[..., 2:4], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
            K1 = torch.tensor(data['K1']) # [3, 3] -- paired with x
            K2 = torch.tensor(data['K2']) # [3, 3] -- paired with y
            x = normalize_points(x, K1)
            y = normalize_points(y, K2)
            C = torch.cat([x,y], dim=-1)
            data['correspondences'] = C.numpy()
            self.padd_data(data)
            return data
        else:
            return self.__getitem__load(index)


from types import SimpleNamespace


if __run__:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    #
    # dataset_info = PhotoTourismRootSIFT
    if False:
        dataset_info = KITTI
        src = 'test'
        dataset = ResidualData(dataset_info, src, padding=True) # padding the 
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
        test_loader(loader)
    if True:
        dataset_info = HEB
        # scene = 'Piazza_del_Popolo'
        scene = 'Alamo'
        dataset = H_dataset(dataset_info, scene, padding=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
        test_loader(loader)
    
# %%

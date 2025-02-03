import os
import sys
from argparse import ArgumentParser

tasks_per_gpu = 2

op = ArgumentParser()
op.add_argument("--ntasks-per-gpu", type=int, default=2, dest="tasks_per_gpu")
op.add_argument("-p", "--partition", type=str, default="amdgpufast")
op.add_argument("--amdgpu", action="store_const", const="amdgpu", dest="partition")
op.add_argument("--amdgpufast", action="store_const", const="amdgpufast", dest="partition")
op.add_argument("--gpufast", action="store_const", const="gpufast", dest="partition")
op.add_argument("--gpu", action="store_const", const="gpu", dest="partition")
op.add_argument("--mem", type=int, default=16, help="memory in GB")
op.add_argument("--test", action="store_true", default=False, help="Test only")

#-------------------------------

def mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if len(sub_path) > 0 and not os.path.exists(sub_path):
        mkdir_recursive(sub_path)
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileExistsError:  # it could have been created in the meantime by a parallel process
            pass


def force_path(file_name):
    mkdir_recursive(os.path.dirname(file_name))


class dotdict(dict):
    __getattr__ = dict.get
    __getitem__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)


class Tee(object):
    def __init__(self, name, mode='wt'):
        self.fname = name
        self.mode = mode

    def __enter__(self):
        self.file = open(self.fname, self.mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, *args):
        self.flush()
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


if __name__ == "__main__":
    (opts, args) = op.parse_known_args()
    if len(args) < 1:
        raise RuntimeError("Require 1 positional argument: jobs file")
    cfgname = args[0]
    with open(cfgname, 'r') as f:
        lines = f.readlines()
    print(f'Reading: {cfgname} -- {len(lines)} lines')
    o = dotdict(**vars(opts))
    # # go through all lines -> skip completed tasks
    # for (i, l) in enumerate(lines):
    #     print(l)
    cfgnoext = os.path.splitext(cfgname)[0]
    jobs = len(lines) // o.tasks_per_gpu
    # create config
    batch_file = cfgnoext + '.batch'
    code = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    logs_path, logs_name = os.path.split(cfgnoext)
    logs = logs_path + '/logs/' + logs_name
    force_path(logs)
    with Tee(batch_file, 'wt'):
        print("#!/bin/bash")
        print("#SBATCH --nodes=1")
        print(f"#SBATCH --ntasks={o.tasks_per_gpu}")
        print(f"#SBATCH --cpus-per-task=2")
        print(f"#SBATCH --gres=gpu:1")
        print(f"#SBATCH --partition={o.partition}")
        print(f"#SBATCH --output={logs}_%a_%t.out")
        print(f"#SBATCH --mem={o.mem}G")
        print(f"#SBATCH --job-name={cfgname}")
        print(f"#SBATCH -D {code}")
        print("ml matplotlib/3.5.2-foss-2022a")
        print("ml torchvision/0.15.0-rc1-foss-2022a-CUDA-11.7.0")
        print(f"srun --ntasks={o.tasks_per_gpu} --gres=gpu:1 --gpu-bind=map_gpu:0 python3 rci_worker.py {cfgname}")
    if o.test:
        # os.system(f'sbatch --array=0-{1} {batch_file}')
        pass
    else:
        os.system(f'sbatch --array=0-{jobs - 1} {batch_file}')

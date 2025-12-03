import os
import sys
import subprocess
# import torch

#----------------------
from depreicated.rci import tasks_per_gpu

keys = ["SLURM_NODEID", "SLURM_ARRAY_TASK_ID", "SLURM_STEP_ID", "SLURM_PROCID", "SLURM_STEP_NUM_TASKS", "SLURM_TASK_PID"]
# print SLURM variables identifying the task 
print("".join([f"{k}={os.environ[k]} " for k in keys]))

if len(sys.argv) < 2:
    raise RuntimeError("Require 1 argument: jobs file")
cfgname = sys.argv[1]
with open(cfgname, 'r') as f:
    lines = f.readlines()
jobs = len(lines) // tasks_per_gpu
# get our job id
job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
task_id = int(os.environ["SLURM_PROCID"])
line_id = job_id * tasks_per_gpu + task_id
if (line_id < len(lines)):
    l = lines[line_id]
    # l = l[3:-1]
    print(f"job_id: {job_id} task_id: {task_id}: Executing: {l}")
    sys.stdout.flush()
    result = os.system(l)

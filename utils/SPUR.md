# slurm



# SUPR

[website](https://supr.snic.se/)

[doc](https://www.c3se.chalmers.se/)

## Connect use SSH

1. Reset your password(for the login account) if you havn't

2. VPN connected to kth network, password: Wv6k_F4n5

3. ssh huizha@alvis1.c3se.chalmers.se  

   password: kth10044!sp


ssh puzhao@alvis1.c3se.chalmers.se 
password: kth10044SUPR!


## Project info 

# Interactive use

https://www.c3se.chalmers.se/documentation/applications/jupyter/

1. Interactive console

   srun -A SNIC2020-33-43 -p alvis -t 07:30:00 --pty --gpus-per-node=V100:1 bash

2. interactive jupyter — gpu, cluster node

   srun -A SNIC2020-33-43 -p alvis -t 07:30:00 --pty --gpus-per-node=V100:1 jupyter notebook
   
3. Jupyter — login node

   1. Load module: `module load GCCcore/10.2.0 IPython/7.18.1`
   2. `jupyter notebook`

==Note: cannot open the notebook link==

# Job 

## Job commands

- `sbatch`: submit batch jobs
- `srun`: submit interactive jobs
- `jobinfo`, `squeue`: view the job-queue and the state of jobs in queue
- `scontrol show job <jobid>`: show details about job, including reasons why it's pending
- `sprio`: show all your pending jobs and their priority
- `scancel`: cancel a running or pending job
- `sinfo`: show status for the partitions (queues): how many nodes are free, how many are down, busy, etc.
- `sacct`: show scheduling information about past jobs
- `projinfo`: show the projects you belong to, including monthly allocation and usage
- For details, refer to the -h flag, man pages, or google!

## Job monitoring

- **top** will show you how much CPU your process is using, how much memory, and more. Tip: press ‘H’ to make top show all threads separately, for multithreaded programs

- **iot p** can show you how much your processes are reading and writing on disk
- Debugging with Allinea Map, **gdb**

## Examples (Puzhao)

## KTH-VPN
vpn.lan.kth.se
- puzhao@kth.se
- bDwcx8kfq

## Connect to Server
ssh puzhao@alvis1.c3se.chalmers.se 
password: kth10044SUPR!

## Move data from local to remote
scp -r E:\SAR4Wildfire_Dataset\Temporal_Progression_Dataset\BC2018R12068_Progression_Data_20m puzhao@alvis1.c3se.chalmers.se:~/Temporal_Progression_Dataset/

scp -r E:\SAR4Wildfire_Dataset\Temporal_Progression_Dataset\BC2018R12068_Progression_Data_20m puzhao@alvis1.c3se.chalmers.se:~/Temporal_Progression_Dataset/

## Move data from remote to local
scp -r puzhao@alvis1.c3se.chalmers.se:~/eo4wildfire_wandb_seed5 E:\SAR4Wildfire_Dataset

scp -r puzhao@alvis1.c3se.chalmers.se:~/eo4wildfire_wandb_TV2 E:\SAR4Wildfire_Dataset

scp -r puzhao@alvis1.c3se.chalmers.se:~/eo4wildfire_wandb/Experiments_Offline /Users/puzhao/Downloads/ESA_SAR4Wildfire/Experiments_Offline

can run directly

1. Copy code from login node


### Submit Job Array
sbatch --array=1-5 GLV1_beta_1e-5.sh
### Check Queue
squeue
### Check output
cat ***.out


## Job Array (Loop Seed: Puzhao)
```shell
#!/bin/bash
#SBATCH -A SNIC2020-33-43
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1 
#SBATCH -t 7-00:00:00
#SBATCH --job-name UNet_OptREF_woCAug
#SBATCH --output UNet_OptREF_woCAug.out

git clone https://github.com/puzhao89/eo4wildfire.git $TMPDIR/eo4wildfire
cd $TMPDIR/eo4wildfire

rsync -a $SLURM_SUBMIT_DIR/Global_SAR4Wildfire_Dataset_V1 $TMPDIR/eo4wildfire/Data/Historical_Wildfire_Dataset/
rsync -a $SLURM_SUBMIT_DIR/Temporal_Progression_Dataset $TMPDIR/eo4wildfire/Data/

ls $TMPDIR/eo4wildfire/Data/Historical_Wildfire_Dataset/

exp_dir=$SLURM_SUBMIT_DIR/eo4wildfire_wandb_TV2
mkdir $exp_dir

while sleep 20m
do
    rsync -a $TMPDIR/eo4wildfire/outputs/Experiments_Offline $exp_dir
done &
LOOPPID=$!

singularity exec --nv $SLURM_SUBMIT_DIR/PyTorch_v1.7.0-py3.sif python run_SegModel_wandb.py data.ref_mode=OptREF data.random_state=$SLURM_ARRAY_TASK_ID model.gamma=1 model.alpha=10 model.beta=0 data.train_val_split_rate=0.5 data.useDataWoCAug=True model.ARCH=UNet data.p_channelAug=0 experiment.exp_name=halfVal_CAug_0

rsync -a $TMPDIR/eo4wildfire/outputs/Experiments_Offline $exp_dir
kill $LOOPPID
```

### Submit Job Array
sbatch --array=1-5 tc-default.sh

## temporal-consistency-job
```shell
#!/bin/bash
#SBATCH -A SNIC2020-33-43
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1 
#SBATCH -t 7-00:00:00
#SBATCH --job-name tc_exp
#SBATCH --output tc_exp.out

git clone https://github.com/puzhao89/temporal-consistency.git $TMPDIR/temporal-consistency
cd $TMPDIR/temporal-consistency

rsync -a $SLURM_SUBMIT_DIR/data_for_snic/data $TMPDIR/temporal-consistency/

ls $TMPDIR/temporal-consistency/data/

exp_dir=$SLURM_SUBMIT_DIR/tc4wildfire_experiments/
mkdir $exp_dir

while sleep 20m
do
    rsync -a $TMPDIR/temporal-consistency/outputs/* $exp_dir
done &
LOOPPID=$!

singularity exec --nv $SLURM_SUBMIT_DIR/PyTorch_v1.7.0-py3.sif python main.py

rsync -a $TMPDIR/temporal-consistency/outputs/* $exp_dir
kill $LOOPPID
```

## Connect to Server
ssh puzhao@alvis1.c3se.chalmers.se 
password: kth10044SUPR!

```shell
#!/bin/bash
#SBATCH -A SNIC2020-33-43
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1 
#SBATCH -t 7-00:00:00
#SBATCH --job-name FCN_SARREF_beta
#SBATCH --output FCN_SARREF_beta.out

git clone https://github.com/puzhao89/eo4wildfire.git $TMPDIR/eo4wildfire
cd $TMPDIR/eo4wildfire

rsync -a $SLURM_SUBMIT_DIR/Global_SAR4Wildfire_Dataset_V1 $TMPDIR/eo4wildfire/Data/Historical_Wildfire_Dataset/
rsync -a $SLURM_SUBMIT_DIR/Temporal_Progression_Dataset $TMPDIR/eo4wildfire/Data/

ls $TMPDIR/eo4wildfire/Data/Historical_Wildfire_Dataset/

exp_dir=$SLURM_SUBMIT_DIR/eo4wildfire_wandb_TV2
mkdir $exp_dir

while sleep 20m
do
    rsync -a $TMPDIR/eo4wildfire/outputs/Experiments_Offline $exp_dir
done &
LOOPPID=$!

for beta in 10 0; do
  singularity exec --nv $SLURM_SUBMIT_DIR/PyTorch_v1.7.0-py3.sif python run_SegModel_wandb.py data.ref_mode=SARREF data.random_state=$SLURM_ARRAY_TASK_ID model.gamma=1 model.alpha=10 model.beta=$beta data.train_val_split_rate=0.5 data.useDataWoCAug=True model.ARCH=FCN model.BATCH_SIZE=16 experiment.exp_name=halfVal
done

rsync -a $TMPDIR/eo4wildfire/outputs/Experiments_Offline $exp_dir
kill $LOOPPID
```

sbatch --array=1-5 DeepLab_beta.sh


```shell
#!/bin/bash
#SBATCH -A SNIC2020-33-43
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1 
#SBATCH -t 7-00:00:00
#SBATCH --job-name GLV1_valMaskAccP
#SBATCH --output GLV1_valMaskAccP.out

git clone https://github.com/puzhao89/eo4wildfire.git $TMPDIR/eo4wildfire
cd $TMPDIR/eo4wildfire

rsync -a $SLURM_SUBMIT_DIR/Global_SAR4Wildfire_Dataset_V1 $TMPDIR/eo4wildfire/Data/Historical_Wildfire_Dataset/
rsync -a $SLURM_SUBMIT_DIR/Temporal_Progression_Dataset $TMPDIR/eo4wildfire/Data/

ls $TMPDIR/eo4wildfire/Data/Historical_Wildfire_Dataset/

exp_dir=$SLURM_SUBMIT_DIR/eo4wildfire_wandb_addDataWoCAug
mkdir $exp_dir

while sleep 20m
do
    rsync -a $TMPDIR/eo4wildfire/outputs/Experiments_Offline $exp_dir
done &
LOOPPID=$!

for trainValSplit in 0.5 0.3; do
  singularity exec --nv $SLURM_SUBMIT_DIR/PyTorch_v1.7.0-py3.sif python run_SegModel_wandb_valMaskAcc.py data.ref_mode=OptSAR data.random_state=$SLURM_ARRAY_TASK_ID model.gamma=1 model.alpha=10 model.beta=0 data.useDataWoCAug=False data.train_val_split_rate=$trainValSplit experiment.exp_name=valMaskAccP_$trainValSplit model.max_epoch=3
done

rsync -a $TMPDIR/eo4wildfire/outputs/Experiments_Offline $exp_dir
kill $LOOPPID
```

sbatch --array=1-5 GLV1_OptREF_g1_a10_b0_addDataWoCAug.sh




## Job Array (HUI)
``` shell
#!/bin/bash
#SBATCH -A SNIC2020-33-43
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1
#SBATCH -t 7-00:00:00
#SBATCH --output noise_bc.out


git clone --single-branch --branch hui https://github.com/Celiali/FixMatch.git $TMPDIR/FixMatch
cd $TMPDIR/FixMatch
mkdir outputs
mkdir checkpoints

exp_dir=$SLURM_SUBMIT_DIR/noise_bc_seed$SLURM_ARRAY_TASK_ID
mkdir $exp_dir

while sleep 6h; 
do
    rsync -a $TMPDIR/FixMatch/outputs $exp_dir
    rsync -a $TMPDIR/FixMatch/checkpoints $exp_dir
    rsync -a $TMPDIR/FixMatch/run.log $exp_dir
done &
LOOPPID=$!

singularity exec --nv $SLURM_SUBMIT_DIR/containers_pytorch-v1.7.0-py3.sif python run.py DATASET.label_num=150 DATASET.strongaugment='RA' Logging.seed=$SLURM_ARRAY_TASK_ID DATASET.add_noisy_label=True EXPERIMENT.batch_balanced=True EXPERIMENT.neg_penalty=False EXPERIMENT.eta_negpenalty=0.1 EXPERIMENT.eta_dynamic=True EXPERIMENT.equal_freq=True

rsync -a $TMPDIR/FixMatch/outputs $exp_dir
rsync -a $TMPDIR/FixMatch/checkpoints $exp_dir
rsync -a $TMPDIR/FixMatch/run.log $exp_dir
kill $LOOPPID
```




2. clone code from github

```shell
#!/bin/bash
#SBATCH -A SNIC2020-33-43
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1
#SBATCH -t 7-00:00:00
#SBATCH --output noise_bc.out


git clone --single-branch --branch hui https://github.com/Celiali/FixMatch.git $TMPDIR/FixMatch
cd $TMPDIR/FixMatch
mkdir outputs
mkdir checkpoints

exp_dir=$SLURM_SUBMIT_DIR/noise_bc_seed$SLURM_ARRAY_TASK_ID
mkdir $exp_dir

while sleep 6h; 
do
    rsync -a $TMPDIR/FixMatch/outputs $exp_dir
    rsync -a $TMPDIR/FixMatch/checkpoints $exp_dir
    rsync -a $TMPDIR/FixMatch/run.log $exp_dir
done &
LOOPPID=$!

singularity exec --nv $SLURM_SUBMIT_DIR/containers_pytorch-v1.7.0-py3.sif python run.py DATASET.label_num=150 DATASET.strongaugment='RA' Logging.seed=$SLURM_ARRAY_TASK_ID DATASET.add_noisy_label=True EXPERIMENT.batch_balanced=True EXPERIMENT.neg_penalty=False EXPERIMENT.eta_negpenalty=0.1 EXPERIMENT.eta_dynamic=True EXPERIMENT.equal_freq=True

rsync -a $TMPDIR/FixMatch/outputs $exp_dir
rsync -a $TMPDIR/FixMatch/checkpoints $exp_dir
rsync -a $TMPDIR/FixMatch/run.log $exp_dir
kill $LOOPPID
```



```shell
#!/bin/bash
#SBATCH -A SNIC2020-33-43
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1  # We're launching 2 nodes with 2 Nvidia T4 GPUs each
#SBATCH -t 7-00:00:00
#SBATCH --job-name clip_ab_both_strong
#SBATCH --output clip_ab_both_strong.out

echo 'FixMatch ablation study: use strong augmentation to generate pseudo lables, TH: 0.95 RA aug, #labels:250'

git clone --single-branch --branch hui https://github.com/Celiali/FixMatch.git $TMPDIR/FixMatch
cd $TMPDIR/FixMatch
mkdir outputs
mkdir checkpoints

exp_dir=$SLURM_SUBMIT_DIR/clip_ab_both_strong
mkdir $exp_dir

while sleep 6h;
do
    rsync -a $TMPDIR/FixMatch/outputs $exp_dir
    rsync -a $TMPDIR/FixMatch/checkpoints $exp_dir
    rsync -a $TMPDIR/FixMatch/run.log $exp_dir
done &
LOOPPID=$!
```



```shell
#!/bin/bash
#SBATCH -A SNIC2020-33-43
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1  # We're launching 2 nodes with 2 Nvidia T4 GPUs each
#SBATCH -t 7-00:00:00
#SBATCH --output barely_ra_level1.out

git clone --single-branch --branch hui git@github.com:Hui9409/DD2412Project.git $TMPDIR/FixMatch
cd $TMPDIR/FixMatch
mkdir outputs
mkdir checkpoints

exp_dir=$SLURM_SUBMIT_DIR/barely_ra_level1
mkdir $exp_dir

while sleep 6h; 
do
    rsync -a $TMPDIR/FixMatch/outputs $exp_dir
    rsync -a $TMPDIR/FixMatch/checkpoints $exp_dir
    rsync -a $TMPDIR/FixMatch/run.log $exp_dir
done &
LOOPPID=$!

singularity exec --nv $SLURM_SUBMIT_DIR/containers_pytorch-v1.7.0-py3.sif python run.py DATASET.label_num=10 DATASET.strongaugment='RA' DATASET.barely=True

rsync -a $TMPDIR/FixMatch/outputs $exp_dir
rsync -a $TMPDIR/FixMatch/checkpoints $exp_dir
rsync -a $TMPDIR/FixMatch/run.log $exp_dir
echo 'barely_ra_level1'
kill $LOOPPID
```



Explanations

```shell
#SBATCH --gpus-per-node=V100:1 # allocates 1 V100 GPU (and 8 cores)
#SBATCH --gpus-per-node=T4:1   # allocates 1 T4 GPU (and 4 cores, but you only pay for 2)
```



```bash
Submitted with sbatch --array=0-50:5 diffusion.sh

#!/bin/bash
#SBATCH -A C3SE2017-1-2
#SBATCH -n 40
#SBATCH -t 2-00:00:00

module load intel/2017a


# Set up new folder, copy the input file there
temperature=$SLURM_ARRAY_TASK_ID
dir=temp_$temperature
mkdir $dir; cd $dir
cp $SNIC_NOBACKUP/base_input.in input.in
# Set the temperature in the input file:
sed -i 's/TEMPERATURE_PLACEHOLDER/$temperature' input.in

mpirun $SNIC_NOBACKUP/software/my_md_tool -f input.in


Here, the array index is used directly as input.
It if turns out that 50 degrees was insufficient, then we could do another run:
sbatch --array=55-80:5 diffusion.sh
```

```shell
Submitted with: sbatch -N 3 -J residual_stress_test run_oofem.sh

#!/bin/bash
#SBATCH -A C3SE507-15-6
#SBATCH -p mob
#SBATCH --ntasks-per-node=20
#SBATCH -t 15:00:00
#SBATCH --gres=ptmpdir:1


module load intel/2017a PETSc

cp $SLURM_JOB_NAME.in $TMPDIR
cd $TMPDIR

mkdir $SLURM_SUBMIT_DIR/$SLURM_JOB_NAME
while sleep 1h; do
  rsync *.vtu *.osf $SLURM_SUBMIT_DIR/$SLURM_JOB_NAME
done &
LOOPPID=$!


echo `date`: Running $SLURM_JOB_NAME
mpirun $HOME/bin/oofem -p -f "$SLURM_JOB_NAME.in"
echo `date`: Finished
kill $LOOPPID
rsync *.vtu *.osf $SLURM_SUBMIT_DIR/oofem/$SLURM_JOBNAME/

```



# Containers

Workflow

![img](https://singularity.lbl.gov/assets/img/diagram/singularity-2.4-flow.png)

Our main repository of HPC and AI/ML containers can be found on the clusters under `/apps/hpc-ai-containers/`.



1. once you are in the container, your home/root directory is bound to your local computer
2. `ctrl+d`退出singularity环境





## Brief usage guide



First, pull the container using the following command:

```shell
singularity pull shub://<image-to-pull>

# Available images to pull: 
c3se/containers:pytorch-v1.7.0-py3
c3se/containers:pytorch-v1.6.0-py3
c3se/containers:pytorch-v1.5.0-py3
```

then use the image in your job submission script:

```
singularity exec PyTorch_vXXX.sif python YOUR-PROGRAM.py
```



\* Running: PyTorch can be imported as a python module:

  ```shell
import torch
print(torch.__version__)
  ```



\* See `/workspace/README.md` inside the container for information on customizing your PyTorch image.



\* For further information see: <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>



## GPU

==!!!To access the GPU, you can use the `--nv` option when running your container, e.g:==

```bash
singularity exec --nv my_image.img  my_gpu_app
```

## Using containers in jobs

Using the image in a job is straight forward, and requires no special steps:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:30:00
#SBATCH -A **your-project** -p hebbe

echo "Outside of singularity, host python version:"
python --version
singularity exec ~/ubuntu.img echo "This is from inside a singularity. Check python version:"
```

## Using modules *inside* your container

If you need to import additional paths into your container using the `SINGULARITYENV_` prefix. This is in particular useful with the `PATH` and `LD_LIBRARY_PATH` which are for technical reasons cleared inside the container environment.

```bash
module load MATLAB
export SINGULARITYENV_PATH=$PATH
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
singularity exec ~/ubuntu.simg matlab -nodesktop -r "disp('hello world');"
```

However, note that is it **very easy** to break other software inside your container by importing the host's `PATH` and `LD_LIBRARY_PATH` into your container. In addition, any system library that the software depends on needs to be installed in your container. E.g. you can not start MATLAB if there is no X11 installed, which is typically not done when setting up a small, lean, Singularity image. Thus, if possible, strive to call modules from outside your container unless you a special need, e.g:

```bash
singularity exec ~/ubuntu.simg run_my_program simulation.inp
module load MATLAB
matlab < post_process_results.m
```



# Intro

The clusters use [Slurm](http://slurm.schedmd.com/). (资源管理系统)

## man-pages

**man** provides documentation for most of the commands available on the system, e.g.

- **man ssh**, to show the man-page for the ssh command
- **man -k** **word**, to list available man-pages containing *word* in the title
-  **man man**, to show the man-page for the man command
- To navigate within the man-pages (same as **less**)
  **space** – to scroll down one screen page
  **b** – to scroll up one screen page
  **q** – to quit from the current man page
  **/** – search (type in word, enter)
  **n** – find next search match (**N** for reverse)
  **h** – to get further help (how to search the man page etc)

## Modules

- Several applications are not available by default, but are available via **modules****
  **[**http://www.c3se.chalmers.se/index.php/Software**](http://www.c3se.chalmers.se/index.php/Software)**

- **To load one or more modules, use the command
  **`module load  module-name [module-name ...]`

  ```shell
  #Example:
  $ python3 --version
  -bash: python3: command not found
  $ module load intel/2016b Python/3.5.2
  $ python3 --version 
  Python 3.5.2
  ```

  `module list` — list currently loaded modules

  `module spider module-name` — search for modules

  `module avail` — show available modules

  `module purge` — unload all current modules

  `module show module-name` — show info about the module

## Storing data

### Important environment variables;

- login node

  - **$SLURM_SUBMIT_DIR**, directory where you submitted your job (available to batch jobs).

  - **$SNIC_BACKUP / \$SNIC_NOBACKUP**, main storage space. Set up automatically. means the network-attached disks on the center storage system will be used 

  - Keep scripts and input files in **$SNIC_BACKUP** and put output in **$SNIC_NOBACKUP**

    **\$SNIC_BACKUP**: 

    space quota — 30GiB 

    Files quota: 60000

-  **Cluster node** — **$TMPDIR**,

   local scratch disk on the node(s) of your jobs. Automatically deleted when the job has finished.

  

- Try to avoid lots of small files: sqlite or HDF5 are easy to use!

- Using sbatch --gres=ptmpdir:1 you get a distributed, parallel $TMPDIR across all nodes in your job. Always recommended for multi-node jobs that use ​\$TMPDIR.

### Check current usage: `C3SE_quota`

```shell
du -h *.* | sort -hr # 查看文件大小并排序
ls -lh ~ # finding where quota is used
ls -a # check all files
singularity cache clean
```





### Download files to local machine

## KTH-VPN
vpn.lan.kth.se
- puzhao@kth.se <br>
- bDwcx8kfq

## Connect to Server
ssh puzhao@alvis1.c3se.chalmers.se <br>
password: kth10044SUPR!

## Move Data
scp -r E:\SAR4Wildfire_Dataset\Temporal_Progression_Dataset\US2020Creek_Progression_Data_20m puzhao@alvis1.c3se.chalmers.se:~/Temporal_Progression_Dataset/

scp -r E:\PyProjects/temporal-consistency\data_for_snic puzhao@alvis1.c3se.chalmers.se:~/data_for_snic/

## unzip
tar -zxvf google-cloud-sdk-339.0.0-linux-x86_64.tar.gz




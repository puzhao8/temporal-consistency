"# temporal-consistency" 


## gcp cp: https://cloud.google.com/storage/docs/gsutil/commands/cp#description
gsutil -m cp -r E:\PyProjects/temporal-consistency\data/elephant_s1_9706x5x5x16x16_clean_100m.npy  gs://eo4wildfire/s1_clean_100m


# SNIC (temporal-consistency-job)

## Move data between SNIC, Cloud, and Local 
### SNIC <--> Local
scp -r E:\SAR4Wildfire_Dataset\Temporal_Progression_Dataset\US2020Creek_Progression_Data_20m puzhao@alvis1.c3se.chalmers.se:~/Temporal_Progression_Dataset/

scp -r E:\PyProjects/temporal-consistency\data_for_snic puzhao@alvis1.c3se.chalmers.se:~/data_for_snic/

### Cloud <--> Local
https://cloud.google.com/storage/docs/gsutil/commands/cp
gsutil cp gs://my-bucket/*.txt ./local
gsutil -m cp -r dir gs://my-bucket


### unzip
tar -zxvf google-cloud-sdk-339.0.0-linux-x86_64.tar.gz


## start VPN connection
vpn.lan.kth.se
- puzhao@kth.se <br>
- bDwcx8kfq

## Connect to Server
ssh puzhao@alvis1.c3se.chalmers.se 
password: kth10044SUPR!


## create .sh file for cmd
nano [tc-dft].sh

### check git clone
!git clone https://[username]:[password]@github.com/[username]/temporal-consistency.git main <br>
git clone https://github.com/puzhao89/temporal-consistency.git

copy the following into [tc_fisrt].sh

```shell (NOT include this line)
#!/bin/bash
#SBATCH -A SNIC2020-33-43
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1 
#SBATCH -t 7-00:00:00
#SBATCH --job-name tc-dft
#SBATCH --output tc-dft.out

git clone https://github.com/puzhao89/temporal-consistency.git $TMPDIR/temporal-consistency
cd $TMPDIR/temporal-consistency
pwd

rsync -a $SLURM_SUBMIT_DIR/data_for_snic/data $TMPDIR/temporal-consistency/

ls $TMPDIR/temporal-consistency/data/

exp_dir=$SLURM_SUBMIT_DIR/tc4wildfire_outputs
mkdir $exp_dir

while sleep 20m
do
    rsync -a $TMPDIR/temporal-consistency/outputs/* $exp_dir
done &
LOOPPID=$!

singularity exec --nv $SLURM_SUBMIT_DIR/PyTorch_v1.7.0-py3.sif python main.py

rsync -a $TMPDIR/temporal-consistency/outputs/* $exp_dir
kill $LOOPPID

(NOT include this line) ```

## submit a job

### Submit Job (single config, single trail)
sbatch tc-dft.sh

### Submit Job Array (same config, 5 trails)
sbatch --array=1-5 tc-dft.sh

## monitor a submited job
### check the queue
squeue

### check output
cat tc-dft.out

### cancel a submited job
scancel [JOBID]




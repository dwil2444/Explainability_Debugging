#!/bin/bash
#SBATCH --job-name=compute_ssim_using_rise
#SBATCH --output=./slurm_output/ssim-%j.log
#SBATCH --ntasks=1
#SBATCH --error=./slurm_output/ssim-%j.err
#SBATCH --exclude=cheetah[01-03],lynx[05-10],cortado[01-10],doppio[01-05],hydro,lynx[08-09],slurm[1-5],optane01,titanx[01-06]
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem-per-cpu=64G

source /etc/profile.d/modules.sh
WORK_DIR="/u/dw3zn/Repos/saliency_calibration/"

# check venv
module add python3-3.8.0

if [ ! -d ${WORK_DIR}/venv ]
then
 echo "No Virtualenv found @ ${WORK_DIR}/venv"
fi
source ${WORK_DIR}/venv/bin/activate

which python3
python3 --version
echo ${HOSTNAME}

python3 -m Temperature_Scaling.compare_models

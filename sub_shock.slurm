#!/bin/bash
#SBATCH -J story_JOBNAME # Request name
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --threads-per-core=1

#SBATCH --constraint=HSW24
#SBATCH --time=23:00:00 
#SBATCH --output Story.output

#SBATCH --mail-type=END
#SBATCH --mail-user=clement.moissard@lpp.polytechnique.fr

cd ${SCRATCHDIR}/JOBNAME/story/

jupyter nbconvert --to python Shock_tracking_save_slices.ipynb

ipython Shock_tracking_save_slices.py 

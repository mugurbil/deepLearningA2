Deep Learning 2015 Spring A2
M&M: Mehmet Ugurbil, Mark Liu

centroid_finder -> centroids.data, feature_means.data, whitening_matrix.data
extract_features -> train_features.data
supervised_learning -> model.data
classify -> ...

-- Run the following pbs on mercer to generate result.csv --
#!/bin/bash

#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
#PBS -l mem=64GB
#PBS -N M_AND_M

cd ~
mkdir M_AND_M
cd M_AND_M/

wget http://cims.nyu.edu/~ml4133/m_and_m_a2.pdf

git clone https://github.com/mugurbil/deepLearningaA2.git m_and_m
cd m_and_m
/scratch/courses/DSGA1008/bin/th classify.lua -start 1 -finish 8000 -label 1 -model /scratch/ml4133/model_epoch17.data
/scratch/courses/DSGA1008/bin/th results.lua > ../predictions.csv

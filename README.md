# MEGC2024-CODE
Code for MECG 2024 task 2: ME spotting and recognition  

We use five-fold cross-validation to split the training set and test set.  
We use a fixed random seed to ensure the reproducibility of the experiments.  
To avoid overfitting, we train only 8 epochs for every fold.  

## Prerequisites

1. Place the `shape_predictor_68_face_landmarks.dat` file in the `model` folder. The file can be downloaded from [this link](https://drive.google.com/file/d/142qa90awRjQbOB_hZSTj6eupSHbcIFY-/view?usp=sharing).
2. Unzip the `casme_feature.zip` file and update the `arg.casme_feature_path` to point to it. The file can be downloaded from [this link](https://drive.google.com/file/d/1GsWF_4wpf1mmpL27QWI8Eu5e6U1osykd/view?usp=sharing).
3. Download the CAS(ME)², CASMEII, SAMM, and SAMM Long Video datasets. Place the labels in the `label/datasetlabel` folder.
4. Modify the 10th column of the `CAS(ME)^2code_final(Updated).csv` file to include the corresponding video path. For example:


1,anger1_1,557,572,608,4+10+14+15,negative,macro-expression,anger,51,s15/15_0401girlcrashing  
1,anger1_2,2854,2862,2871,38,others,macro-expression,sadness,17,s15/15_0401girlcrashing

## Running the Script

Run `main_spotandrecog.py` for micro-expression analysis.

By default, preliminary spotting results are saved in the `label/spot_result_test` folder. The final spotting and recognition results are saved in the `label/spot_recog_test` folder. Before running main_spotandrecog.py, ensure that the CSV file (cas_pred.csv and samm_pred.csv) contain only the header. Use the 'train_ME_recog_network.py' to train recognition module.



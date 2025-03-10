# CS-433: Machine Learning Fall 2023, Project 
#
- Team Name: Clementines
- Team Member:
    1. Lou Fourneaux, **SCIPER: 311084** (lou.fourneaux@epfl.ch)
    2. Adrien Feillard, **SCIPER: 315921** (adrien.feillard@epfl.ch)
    3. Laissy Aurélien, **SCIPER: 329573** (aurelien.laissy@epfl.ch)

* [Getting started](#getting-started)
    * [Project description](#project-description)
    * [Data](#data)
    * [Report](#report)
* [Reproduce results](#reproduce-results)
    * [Requirements](#Requirements)
    * [Results](#results)
    * [Repo Architecture](#repo-architecture)
    

# Getting started
## Project description
This project's primary objective is to determine the accuracy of markless systems at detecting joints when performing exercises. Markless pose estimation involves determining the position and orientation of body joints without the need for physical markers. Marker based systems like Vicon are very accurate but can be costly and cannot be used on a daily life basis. Markless pose estimation has the potential to enhance home-based physiotherapy by providing an affordable and user-friendly solution. This could encourage regular exercise and monitoring for patients in the comfort of their homes. 
More details about the project are available in `project2_description.pdf`.

## Data
The data set was created by recording 25 unimpaired participants doing 3 sets of 7 different exercises among which on is correct and 2 are incorrect. It is composed of about 2.2 million frames captured by 4 webcams. The reference frame is chosen at the waist of the participants. Every frame of the dataset includes the following information:
- The participant's ID
- The type of exercise performed
- Correctness of the exercise: correct or mistake type
- The camera that took this frame
- The time frame in seconds (every 0.033s)
- The position of 33 joints in the X, Y, and Z planes

## Report
All the details about the choices that have been made and the methodology used throughout this project are available in `report.pdf`. In this report, the different asumptions, decisions and results made and found are explained as well as our results.

To reproduce the results the files should be added to the repo, as described in [Repo Architecture](#repo-architecture). 

## Reproduce results
- Download the folder and extract the zip file
- Make sure that all required libraries are properly installed and that the folder respect the repository architecture
- Open a terminal or anaconda prompt and access to the folder where the project folder is located
- (For anaconda activate the environement where all libraries have been installed by typing "conda activate <environement_name>")
- Type python run.py
- You will choose the task you want to either tune, train and or test with "Enter the task number (1 or 2), 'exit' or ctrl+C to quit:"
    - If you type "1" The exercise classification problem (1) will be done.
    - If you type "2" the exercise and mistake classification problem (2) will be done.
- You will choose the model you want to use with "Enter the model (NN, GRU, RF, GBC, CNN, Ethical), 'exit' to come back to the task choice:" 
- You will choose if you want to use a model partially or fully processed with "Do you want to use a saved model? Enter 'yes' or 'no'. If you have already existing model files 'no' will create new ones and destroy the existing ones (If you want to keep the already existing ones change the names of the existing files):" (you can type either in upper case or lower case)
    - If you type 'no' by mistake you can do ctrl+C at anytime before the programm finishes the hyperparameters tuning and it will not destroy the already existing model tuned file.
    - If you want to tune the hyperparameters of a model with different settings and want to keep the files that are already existing make sure to change the name of those files. The newly created files will replace any files with the same name.
    - If you have changed to name of a model file to avoid destroying and want to reuse it make sure to change the name file to "entire_model<model_name>_task<task_number>_tuned.pth or .joblib" or "entire_model<model_name>_task<task_number>_tuned_trained.pth or .joblib"
- If you type 'no' a new model will be created.
- If you type 'yes' you will be asked which model file you want to use with "Choose the save type of the model you want to use: 't' for model tuned with the best parameters or 'tt' for model tuned  and trained with best parameters:"  (you can type either in upper case or lower case)
    - If you type 't' you will be using an already tuned model that will perform training and testing
    - If you type 'tt' you will be using an already tuned and trained that will perform only testing.
- After the model has been test it asks you again all those steps until you want to stop using the programm by typing 'exit' or by doing ctrl+C:

**All of our saved models were trained and saved using this repository architecture which couln't be changed aftewards in order to be able to use the torch and joblib saved models. The model.py files and saved models have to be in this architecture in order for the saved models to be recoverable. Moving them could corrupt them**

After every training a file 'training_result_<model_name>_<task_number>.csv' will be created saving all the history of the epoch, loss and accuracy of the training.
After every testing a file 'test_result_<model_name>_<task_number>.csv' will be created saving all the true encoded labels and the predicted ones.

To create the accuracy plot of the different neural networks and the confusion matrices of the model tests you just need to type "exit" to quit the programm and the plots will be automatically saved inside the folder results as the files accuracy_plot.png and confusion_matrices.png and the values of the confusion matrices are saved inside the folder results/results csv <model_name>_task<task_number>.csv_raw_data.csv

You do not need to run every task and every model to get the plots. It plots the data coming the .csv files from the results/results csv folder 
**All newly created files will replace the already existing ones. Rename the files you don't want to update to keep them and rename them accordingly if you wish to use them in the programm**


**Side note on Ethical model:** This model can only be processed by taking the second task. If you choose task 1 then ethical it will not work. 

**Side note on Random Forest results:** The test_result_rf_1.csv test_result_rf_2.csv and training_result_rf_1.csv training_result_rf_2.csv files for random forest on task 1 and 2 got corrupted. So if you want to reproduce the results you need to retrain them.

## Requirements
- Python==3.11.5
- Libraries (Version number only indicates the version of the library under which the project has been run. Older libraries versions might not work):
    - scikit-learn==1.3.0
    - matplotlib==3.8.0
    - numpy==1.24.3
    - pandas==2.1.1
    - seaborn==0.12.2
    - pytorch==2.1.0
    - skorch==0.15.0
    - pyarrow==11.0.0

Install those libraries by typing "pip install <library>" on the terminal or "conda install <library>" after activating the desired environnement if working on a anaconda prompt.
The files : entire_modelrf_task1_tuned_trained.joblib, entire_modelrf_task2_tuned_trained.joblib and All_Relative_Results_Cleaned.parquet are not available in this github repository. (Downloading the files are impossible as they are private)
Put the files in the project's folder following the repository architecture. Replace the empty file in the Dataset folder by All_Relative_Results_Cleaned.parquet

## Results
The accuracies of each model are summarized in the report with the confusion matrices.

## Repository architecture
<pre>
├── README.md
├── __init__.py
├── run.py
├── data_processing.py
├── neural_network_model.py
├── GRU_neural_network.py
├── convolutional_neural_network.py
├── random_forest_classifier.py
├── entire_modelnn_task1_tuned.pth
├── entire_modelnn_task1_tuned_trained.pth
├── entire_modelnn_task2_tuned.pth
├── entire_modelnn_task2_tuned_trained.pth
├── entire_model_gru_task1_tuned.pth
├── entire_model_gru_task1_tuned_trained.pth
├── entire_model_gru_task2_tuned.pth
├── entire_model_gru_task2_tuned_trained.pth
├── entire_modelrf_task1_tuned.joblib
├── entire_modelrf_task1_tuned_trained.joblib
├── entire_modelrf_task2_tuned.joblib
├── entire_modelrf_task2_tuned_trained.joblib
├── entire_modelcnn_task1_tuned.pth
├── entire_modelcnn_task1_tuned_trained.pth
├── entire_modelcnn_task2_tuned.pth
├── entire_modelcnn_task2_tuned_trained.pth
├── dataset
│   └── All_Relative_Results_Cleaned.parquet
├── project_description
│    ├── project2_description.pdf
│    └── report.pdf
└── results
    ├── accuracy_plot.png
    ├── confusion_matrices.png
    ├── confusion_matrix_nn_2_ethical.png
    ├── visualization.py
    └── results_csv
        ├── test
        │    ├── cnn_task1_raw_data.csv
        │    ├── cnn_task2_raw_data.csv
        │    ├── gru_task1_raw_data.csv
        │    ├── gru_task2_raw_data.csv
        │    ├── nn_task1_raw_data.csv
        │    ├── nn_task2_raw_data.csv
        │    ├── test_result_nn_1.csv
        │    ├── test_result_cnn_1.csv
        │    ├── test_result_gru_1.csv
        │    ├── test_result_rf_1.csv
        │    ├── test_result_nn_2.csv
        │    ├── test_result_cnn_2.csv
        │    ├── test_result_gru_2.csv
        │    ├──test_result_rf_2.csv
        │    └── test_result_nn_2_ethical.csv
        └── train
             ├──training_result_nn_1.csv 
             ├──training_result_cnn_1.csv 
             ├──training_result_gru_1.csv 
             ├──training_result_nn_2.csv 
             ├──training_result_cnn_2.csv 
             └──training_result_gru_2.csv 
    
    
             

</pre>

# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Udacity first project is concerned about predicting the Churn for clients given their status and features.
It is a classification project which the main steps are the following ones:

- 1°) Importation of the archives;
- 2°) Performing the EDA of the data;
- 3°) Preprocessing categorical features;
- 4°) Training models (Random Forest Classifier and Logistic Regression);
- 5°) Evaluating the models trained;
- 6°) Tracking of most relevant features;
- 7°) Saving the final models.

## Files and data description
Overview of the files and data present in the root directory.

**Folders**
- data: csv file with the data of the project

- images
  - eda: churn, customer age, marital status and total transactions distribution and heatmap;
  - results: feature importance, logistic and random forest results, roc curve.

- logs: file with the test logs through the script "churn_script_logging_and_tests.py";

- models: saved models of logistic regression and random forest classifier.

**Files**
- churn_library.py: file that process the whole models;
- churn_notebook.ipynb: same file as "churn_library.py", but applied into a notebook;
- churn_script_logging_and_tests.py: process the pytest and log the events;
- Guide.ipynb: general guidance for the project;
- README.md: folder with general explanations and overview of the projects;
- requirements_py3.6.txt: requierements for py36 that needs to be installed;
- requirements_py3.8.txt: requierements for py38 that needs to be installed.


## Running Files
*How do you run your files? What should happen when you run your files?*
  - There are two main approaches as mention before, either the "churn_library.py" can be run as a single file, or command by command through "churn_notebook.ipynb";
  - In order for the execution to un properly, the requierements.txt needs to be installed, so every library that it is needed it is installed;
  - When running the "churn_library.py" few archives will be generated at the folders images and models:
    - images:
      - eda: total_transaction_distribution.png, marital_status_distribution.png, heatmap.png, customer_age_distribution.png and churn_distribution.png;
      - results: feature_importances.png, rf_results.png, logistic_results.png and roc_curve_results.png.
    - models: logistic_model.pkl and rfc_model.pkl.
  - File "churn_script_logging_and_tests.py" is used to evaluate the integraty of the script "churn_library.py". The command "pytest churn_script_logging_and_tests.py" is called for testing it;
    - logs: calling this command generates a log file in the folder logs called "churn_library.log" that contains the results and logging of the test before.

## Evaluation of the Models
By the metrics reported in the results folder (inside images) it is possible to track both models.
- Logistic regression

![image](https://github.com/VD-git/CleanCode/assets/85261454/bfd12631-77bb-4cff-84a9-60ded437d90a)
- Random forest classifier

![image](https://github.com/VD-git/CleanCode/assets/85261454/3f99c863-2ee7-4b44-af1c-811b446b38cc)

Besides that it is also possible to evalute the roc curve and feature importance (taking into consideration random forest classifier)
- Roc curve



- Feature importance






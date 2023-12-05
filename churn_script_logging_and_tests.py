"""
File that does the testing and logging for the file "churn_library.py"

Author: Victor Dias
Creation Date: 05/12/2023
"""

import os
import logging
import pytest
import churn_library as cls

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.mark.parametrize("filename", ["./data/bank_data.csv"])
def test_import(filename):
    '''
    test data import - this example is completed for you to assist with
    the other test functions
    '''
    # Evaluating importation of the file
    try:
        pytest.df = cls.import_data(filename)
        logging.info("[import_data] SUCCESS: File path %s found", filename)
    except FileNotFoundError as err:
        logging.error("[import_data] ERROR: File path %s not found", filename)
        raise err

    # Evaluating shape of the data imported
    try:
        assert pytest.df.shape[0] > 0
        assert pytest.df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "[import_data] ERROR: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    # Performing the EDA without graphs
    try:
        logging.info("[perform_eda] SUCCESS: Shape of the file imported: %s",
                     pytest.df.shape)
        if sum(pytest.df.isnull().sum() > 0):
            logging.warning(
                "[perform_eda] WARNING: Missing data found:\n%s",
                pytest.df.isnull().sum())
        else:
            logging.info("[perform_eda] SUCCESS: No missing found")
        logging.info("[perform_eda] SUCCESS: Summary of the data:\n%s",
                     pytest.df.describe())
    except Exception as err:
        logging.error(
            "[perform_eda] ERROR: Failed to perform the EDA - %s", err)

    # Performing the EDA with graphs
    try:
        cls.perform_eda(pytest.df)
        logging.info("[perform_eda] SUCCESS: All plots have been saved")
    except KeyError as err:
        logging.error(
            "[perform_eda] ERROR: Failed to plot the EDA graphs - %s", err)


@pytest.mark.parametrize("category_lst",
                         [['Gender',
                           'Education_Level',
                           'Marital_Status',
                           'Income_Category',
                           'Card_Category']])
def test_encoder_helper(category_lst):
    '''
    test encoder helper
    '''
    try:
        for i in category_lst:
            assert i in pytest.df.columns
        cls.encoder_helper(pytest.df, category_lst, "Churn")
        logging.info(
            "[encoder_helper] SUCCESS: All %s cols have been encoded",
            len(category_lst))
    except AssertionError as err:
        logging.error(
            "[encoder_helper] ERROR: Encoding not successful - %s", err)


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        (pytest.X_train, pytest.X_test, pytest.y_train, pytest.y_test) = \
            cls.perform_feature_engineering(pytest.df, "Churn")
        logging.info(
            "[feature_engineering] SUCCESS: Train size: %s rows and Test size: %s rows", len(
                pytest.X_train), len(
                pytest.X_test))
    except KeyError as err:
        logging.error(
            "[feature_engineering] ERROR: Split not implemented - %s", err)


def test_train_models():
    '''
    test train_models
    '''
    try:
        cls.train_models(
            pytest.X_train,
            pytest.X_test,
            pytest.y_train,
            pytest.y_test)
        logging.info(
            "[training_models] SUCCESS: Models trained in the proper way")
    except Exception as err:
        logging.error(
            "[training_models] ERROR: Models not trained correctly - %s", err)


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

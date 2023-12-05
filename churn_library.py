# library doc string
"""
File that train both models (logistic regression and random forest classifier),
saving them in the end;
Further than that, it generates evaluation metrics and plots for the models and EDA.
"""

# import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

quant_columns = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]


def save_eda_plots(col, fig_name, graph_type, df):
    '''
    returns the plot of EDA analysis

    input:
            col: column that will be plot
            fig_name: name of the plot that will be saved
            graph_type: type of graph (hist, value_counts, histplot, heatmap)
            df: pandas dataframe
    output:
            fig: plot of the graph saved in ./images/eda/
    '''
    assert graph_type in ['hist', 'value_counts',
                          'histplot', 'heatmap'], "Select a valid plot"

    fig = plt.figure(figsize=(20, 10))
    if graph_type == 'hist':
        df[col].hist()
    elif graph_type == 'value_counts':
        df[col].value_counts('normalize').plot(kind='bar')
    elif graph_type == 'histplot':
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    elif graph_type == 'heatmap':
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig.savefig(f'./images/eda/{fig_name}.png')
    plt.close()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df.drop(['Attrition_Flag'], axis=1, inplace=True)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    save_eda_plots(
        col='Churn',
        fig_name='churn_distribution',
        graph_type='hist',
        df=df)
    save_eda_plots(
        col='Customer_Age',
        fig_name='customer_age_distribution',
        graph_type='hist',
        df=df)
    save_eda_plots(
        col='Marital_Status',
        fig_name='marital_status_distribution',
        graph_type='value_counts',
        df=df)
    save_eda_plots(
        col='Total_Trans_Ct',
        fig_name='total_transaction_distribution',
        graph_type='histplot',
        df=df)
    save_eda_plots(col=None, fig_name='heatmap', graph_type='heatmap', df=df)


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    new_col_list = []
    for i in category_lst:
        mapping_dict = df.groupby(i).mean()['Churn'].to_dict()
        new_col_name = "_".join([str(i), str(response)])
        df[new_col_name] = df[i].apply(lambda x: mapping_dict.get(x))
        new_col_list.append(new_col_name)
        df.drop(i, axis=1, inplace=True)

    return df[new_col_list]


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument,
              that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = df.drop([response], axis=1)
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return (X_train, X_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results,
    and stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.00, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.35, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.70, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close()

    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.00, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.35, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.70, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # refit = True, best model is kept as default model when calling predict
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        refit=True)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Training the different models available
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_train_preds_rfc = cv_rfc.predict(X_train)
    y_test_preds_rfc = cv_rfc.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Saving the classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rfc,
                                y_test_preds_lr,
                                y_test_preds_rfc)

    # Plot the feature importance curve
    feature_importance_plot(cv_rfc,
                            X_train,
                            './images/results/feature_importances.png')

    # Plot the ROC-curve
    fig, ax = plt.subplots(1, 2)
    plot_roc_curve(lrc, X_test, y_test, ax=ax[0])
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax[1])
    fig.set_figwidth(15)
    fig.set_figheight(8)
    fig.savefig('./images/results/roc_curve_results.png')
    plt.close()

    # Saving the models into the folder
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':
    df = import_data(pth=r"./data/bank_data.csv")
    perform_eda(df=df)
    df_encoded = encoder_helper(
        df=df,
        category_lst=cat_columns,
        response='Churn')
    df = pd.concat([df[quant_columns], df_encoded, df['Churn']], axis=1)
    X_train, X_test, y_train, y_test =\
        perform_feature_engineering(df=df, response='Churn')
    train_models(X_train, X_test, y_train, y_test)

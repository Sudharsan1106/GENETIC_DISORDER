# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 11:05:19 2021

@author: sudsa
"""

import mlcp.pipeline as pl
import mlcp.classifires as cl
import mlcp.regressors as rg
import numpy as np
import pandas as pd
from datetime import datetime as dt
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from skmultilearn.adapt import MLkNN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")
# execution controls
classification = 1  # if regression = 0, classification = 1
read = 1
sampling = 0
primary_analysis = 1  # dev only
visual_analysis = 0
observed_corrections = 1
feature_engineering = 0
analyze_missing_values = 1
treat_missing_values = 1
define_variables = 1
analyze_stats = 0  # dev only
analyzed_corrections = 0
gaussian_transform = 0
polynomial_transform = 0
skew_corrections = 0
scaling = 0  # do for continuous numerical values, don't for binary/ordinal/one-hot
encoding = 1
matrix_corrections = 0
oversample = 1  # dev only
reduce_dim = 0
compare = 0  # dev only
cross_validate = 0  # dev only
grid_search = 0  # dev only
train_classification = 1
uniquevalues = 0
disorder_subclass = 0
final = 0


if read == 1:
    filepath = "data/Dataset/train_genetic_disorders.csv"
    y_name = "Genetic Disorder"
    #y_name = ['Genetic Disorder','Disorder Subclass']
    dtype_file = "genetic_sample_dtype_analysis.txt"
    df = pl.read_data(filepath)
    pass
    if classification == 1:

        # for x in y_name:
        sample_diff, min_y, max_y = pl.bias_analysis(df, y_name)
        print("sample diff:", sample_diff)
        print("sample ratio:", min_y/max_y)
        print(df[y_name].value_counts())
        print('\n')

        # print(df[y_name].value_counts())
    else:
        print("y skew--->", df[y_name].skew())

if sampling == 1:
    n_samples = 500
    df = pl.stratified_sample(df, y_name, n_samples)
    print("stratified sample:")
    print(df[y_name].value_counts())


if primary_analysis == 1:
    # consider: unwanted features, numerical conversions (year to no. years),
    # wrong dtypes, missing values, categorical to ordinal numbers
    df_h = df.head()
    with open(dtype_file, "w") as f:
        for c in df_h:
            line1 = df_h[c]
            line2 = df[c].nunique()
            line3 = df[c].isnull().sum()
            f.write(str(line1) + "\n" + "Unique: " + str(line2) +
                    ", missing: " + str(line3)
                    + "\n\n" + "-----------------"+"\n")
    if classification == 0:
        plt.boxplot(df[y_name])
        plt.show()

if uniquevalues == 1:
    for co in df.columns:
        if df[co].dtype != 'object':
            pass
        else:
            print(co.center(60, '='))
            print(df[co].unique())
            print()

if visual_analysis == 1:
    pl.visualize_y_vs_x(df, y_name)


if observed_corrections == 1:
    df = df.drop(['Patient Id', 'Patient First Name', "Father's name",
                  'Location of Institute', 'Institute Name', 'Parental consent', 'Place of birth'], axis=1)
    df['Respiratory Rate (breaths/min)'] = df['Respiratory Rate (breaths/min)'].replace(
        {'Normal (30-60)': 'Normal'})
    df['Birth asphyxia'] = df['Birth asphyxia'].replace(
        {'No record': 'Not available'})
    df['Autopsy shows birth defect (if applicable)'] = df['Autopsy shows birth defect (if applicable)'].replace({
        'None': 'Not applicable'})
    df['H/O radiation exposure (x-ray)'] = df['H/O radiation exposure (x-ray)'].replace(
        {'-': 'No'})
    df['H/O substance abuse'] = df['H/O substance abuse'].replace({'-': 'No'})


if feature_engineering == 1:
    # subjective and optional - True enables the execution
    print("")
    print("Feature Engineering---->:")
    print(df.head())
    pl.visualize_y_vs_x(df, y_name)


if analyze_missing_values == 1:
    drop_th = 0.4
    print(df.shape)
    df = pl.missing_value_analysis(df, drop_th)
    print(df.shape)
    before = len(df)
    df_copy_drop = df.dropna()
    after = len(df_copy_drop)
    print("dropped %--->", round(1-(after/before), 2)*100, "%")
    num_df = df.select_dtypes(exclude=['O'])


if treat_missing_values == 1:
    if set(['Disorder Subclass', 'Genetic Disorder']).issubset(df.columns):
        df.dropna(subset=["Genetic Disorder", "Disorder Subclass"], axis=0)
    for c in df.columns:
        if df[c].dtype != 'O':
            df[c] = df[c].fillna(np.mean(df[c]))

        else:
            c_mode = df[c].value_counts().index[0]
            df[c] = df[c].fillna(c_mode)
    pass


if define_variables == 1:
    y = df['Genetic Disorder']
    x = df.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
    n_dim = x.shape[1]
    print(x.shape)


if analyze_stats == 1:
    # find important features and remove correlated features based on low-variance or High-skew
    cors = pl.correlations(x)
    with open(dtype_file, "a") as f:
        f.write("\n\n\n"+str(cors))
    scores = pl.feature_analysis(x, y)
    print(scores)
    print("")
#    if classification == 1:
#        ranks = pl.feature_selection(x,y); print(ranks); print("")
    for c in x.columns:
        sd, minv, maxv = pl.bias_analysis(x, c)
        print(c, " = ", sd)
    print("")
    print("skew in feature:")
    print(x.skew())


if analyzed_corrections == 1:
    pass


if polynomial_transform == 1:
    degree = 3
    x = pl.polynomial_features(x, degree)
    print("polynomial features:")
    print(x.head(1))
    print("")


if gaussian_transform == 1:
    n_dim = 3
    x = pl.gaussian_features(x, y, n_dim)
    print("Gaussian features:")
    print(x.head(1))
    print(x.shape, y.shape)
    print("")


if skew_corrections == 1:
    x = pl.skew_correction(x)


if scaling == 1:
    selective_scaling = 0

    x_num, x_cat = pl.split_num_cat(x)

    if selective_scaling == 1:
        selective_features = []
        selective_x_num = x_num[selective_features]
        x_num = x_num.drop(selective_features, axis=1)
    else:
        selective_x_num = x_num

    if False:
        selective_x_num, fm = pl.max_normalization(selective_x_num)  # 0-1
    if True:
        selective_x_num = pl.minmax_normalization(selective_x_num)  # 0-1
    if False:
        selective_x_num = pl.Standardization(selective_x_num)  # -1 to 1

    print("")
    print("after scaling - categorical-->", x_cat.info())
    print("after scaling - numerical-->", x_num.shape)

    if selective_scaling == 1:
        x_num = pl.join_num_cat(x_num, selective_x_num)
    else:
        x_num = selective_x_num

    x = pl.join_num_cat(x_num, x_cat)


if encoding == 1:
    x_num, x_cat = pl.split_num_cat(x)

    if True:
        x_cat = pl.label_encode(x_cat)
    if False:
        x_cat = pl.onehot_encode(x_cat)

    if False:
        # best choice if dtypes are fixed
        x, y, mmd = pl.auto_transform_data(x, y)

    x = pl.join_num_cat(x_num, x_cat)
    print("after encoding--->", x.shape)
    print(x.info())


if matrix_corrections == 1:
    x = pl.matrix_correction(x)


if oversample == 1:
    # for only imbalanced data
    x, y = pl.oversampling(x, y)
    print(x.shape)
    print(y.value_counts())


if reduce_dim == 1:
    x = pl.reduce_dimensions(x, 25)  # print(x.shape)
    x = pd.DataFrame(x)
    print("transformed x:")
    print(x.shape)
    print("")


if compare == 1:

    # compare models on sample
    n_samples = 500
    df_temp = pd.concat((x, y), axis=1)
    df_sample = pl.stratified_sample(df_temp, y_name, n_samples)
    print("stratified sample:")
    print(df_sample[y_name].value_counts())
    y_sample = df_sample[y_name]
    x_sample = df_sample.drop([y_name], axis=1)
    x_sample = x_sample.astype(float)
    model_meta_data = pl.compare_models(x_sample, y_sample, 111)


if cross_validate == 1:
    # deciding the random state
    best_model = cl.GradientBoostingClassifier()
    pl.kfold_cross_validate(best_model, x, y, 100)


if grid_search == 1:
    # grids
    dtc_param_grid = {"criterion": ["gini", "entropy"],
                      "max_depth": [2, 3, 4, 6, 8, 10],
                      # "class_weight":[{0:1,1:1.5}],
                      "min_child_weight": [1, 3, 4, 5, 7],
                      "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
                      "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
                      "random_state": [21, 111]
                      }

    log_param_grid = {"penalty": ['l1', 'l2', 'elasticnet'],
                      "C": [0.1, 0.5, 1, 2, 5, 10],
                      "class_weight": [{0: 1, 1: 1}],
                      "solver": ['liblinear', 'sag', 'saga'],
                      "max_iter": [100, 150, 200, 300],
                      "random_state": [100, 111]
                      }

    # print(x.info());

    param_grid = dtc_param_grid
    model = cl.XGBClassifier()
    # a=x.astype(int)
    y = pl.label_encode(y)
    best_param_model = pl.select_best_parameters(model, param_grid, x, y, 30)


if train_classification == 1:
    print(x.shape)
    remove_features = []
#   remove_features=[ 33, 10, 3, 20, 1, 25, 2, 30, 0, 9, 26, 4, 5, 6, 7, 8]; #ohe
    # remove_features=[ 2, 9, 25, 4, 5, 6, 7, 8, 3, 29]; #label

    if remove_features != []:
        x = x.drop(remove_features, axis=1)

#   best_param_model = MLkNN(k=12)
#   trained_model = pl.clf_train_test(best_param_model,x,y,111,"Mlknn")
    x = x.astype(float)
    print("test")
    best_param_model = cl.XGBClassifier(
        random_state=111, max_depth=4, learning_rate=0.1)
    #best_param_model = cl.GradientBoostingClassifier(random_state=195)
    print(y.shape)
    cors = pl.correlations(x)
    print(len(cors))
    #a = y.replace({'Single-gene inheritance diseases':
     #              'Mitochondrial genetic inheritance disorders'})
  #  cors = pl.correlations(a)
    Genetic_model = pl.clf_train_test(best_param_model, x, y, 103, "XGBC1")
    #Genetic_model = pl.kfold_cross_validate(best_param_model,x,y,123)
    if False:
        # recycle the models with most important features
        fi = Genetic_model.feature_importances_
        print("fi count-->", len(fi))
        fi_dict = {}
        for i in range(len(x.columns)):
            fi_dict[x.columns[i]] = fi[i]
        fi_dict = pl.sort_by_value(fi_dict)
        print([k for k, v in fi_dict])


if disorder_subclass == 1:

    y_name = "Disorder Subclass"
    sample_diff, min_y, max_y = pl.bias_analysis(df, y_name)
    print("sample diff:", sample_diff)
    print("sample ratio:", min_y/max_y)
    print(df[y_name].value_counts())
    print('\n')

    # sample difference between subclass X is lesser than former
    y = df['Disorder Subclass']
    x = df.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1)
    n_dim = x.shape[1]
    print(x.shape)

    x_num, x_cat = pl.split_num_cat(x)
    x_cat = pl.label_encode(x_cat)
    x = pl.join_num_cat(x_num, x_cat)
    print("after encoding--->", x.shape)

    x, y = pl.oversampling(x, y)
    print(x.shape)
    print(y.value_counts())

    dtc_param_grid = {"criterion": ["gini", "entropy"],
                      "max_depth": [2, 3, 4, 6, 8, 10],
                      # "class_weight":[{0:1,1:1.5}],
                      "min_child_weight": [1, 3, 4, 5, 7],
                      "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
                      "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
                      "random_state": [21, 111]
                      }

    # param_grid = dtc_param_grid
    # model = cl.XGBClassifier()
    # x=x.astype(int)
    # y=pl.label_encode(y)
    # best_param_model = pl.select_best_parameters(model, param_grid, x, y, 30)

    x = x.astype(int)
    best_param_model = cl.XGBClassifier(
        random_state=100, max_depth=4, gamma=0.4, learning_rate=0.5)
    #best_param_model = cl.GradientBoostingClassifier(random_state=100)
    disorder_model = pl.clf_train_test(best_param_model, x, y, 111, "XGBC1")

if final == 1:
    df1 = pd.DataFrame()
    test_data = pd.read_csv('testdata_cleaned.csv')
    # print(Genetic_model.predict(test_data))
    # print(disorder_model.predict(test_data))
    df1["gene"] = Genetic_model.predict(test_data)
    df1["disorder"] = disorder_model.predict(test_data)
    # print(df.shape())
    print(df1['gene'].value_counts())
    print(df1['disorder'].value_counts())
    pl.visualize_final(df1, 'gene')
    pl.visualize_final(df1, 'disorder')

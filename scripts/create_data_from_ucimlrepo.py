import pandas as pd
import requests
import os
import json
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def create_data_from_ucimlrepo(id):
    # fetch dataset from UCI ML repo
    repo = fetch_ucirepo(id=id) 

    # get metadata as a dictionary
    meta = repo.metadata

    # get basic info from metadata
    name = meta.name.lower().replace(" ", "_")
    task = str.lower(meta.tasks[0])
    target_name = meta.target_col[0]

    #make filepath
    filepath = f'../data/{task}/{name}'

    # create a folder for the dataset with filepath
    os.makedirs(filepath, exist_ok=True)

    # save metadata as a json file
    with open(filepath + '/metadata.json', 'w') as f:
        json.dump(meta, f)

    # get data (as pandas dataframes) 
    X = repo.data.features 
    y = repo.data.targets 

    # make a dictionary mapping col names to Xi and Y
    col_map = {X.columns[i]: f'X{i+1}' for i in range(len(X.columns))}
    col_map[target_name] = 'Y'

    # add a column with new column names to the variables dataframe
    vars = repo.variables
    vars['new_col_name'] = vars['name'].map(lambda x: col_map.get(x, x))

    # Save the variable info as a csv file named data_description.csv
    repo.variables.to_csv(filepath + '/data_description.csv', index=False)

    #rename the columns of X and y using the col_map
    X = X.rename(columns=col_map)
    y = y.rename(columns=col_map)

    # split data, use random state 42 for reproducibility
    # to preserve target distribution, stratify on y if classification, else on deciles of y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if task == 'classification' else pd.qcut(y['Y'], q=10, labels=False))

    # join train sets and test sets into dataframes df_train and df_test
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    #save df_train and X_test as csv files named train.csv and test.csv
    df_train.to_csv(filepath + '/train.csv', index=False)
    df_test.to_csv(filepath + f'/test.csv', index=False)
    # X_test.to_csv(filepath + '/test.csv', index=False)

    # save df_test as a csv file named name_test_private.csv; will be deleted from repo, copy maintained locally by arvind
    # df_test.to_csv(filepath + f'/{name}_test_private.csv', index=False)

idx = [53, # iris
         477, # real estate evaluation
         165, # concrete compressive strength
         492, # metro interstate traffic volume
         109, # wine
         19,  # car evaluation
         101, # tic tac toe endgame
         887  # National Health and Nutrition Examination Survey
        ]

for id in idx:
    create_data_from_ucimlrepo(id=id)
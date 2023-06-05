import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightfm import LightFM
from scipy.sparse import coo_matrix
from scipy.special import expit
import os
import random


class FMScoreWrapper(LightFM):
    def score(self, X, y=None):
        if y is not None:
            preds = self.predict(X.row, X.col)
            # Negative MSE for optimization
            return -mean_squared_error(y, preds)
        else:
            return 0  # Return 0 when y is None


def model_simulation(directory_path, selected_date, category="normal"):
    '''
    Parameters
    ----------
    directory_path (str): Path of the directory of files with the data. FMs will be outputted there.
    selected_date (int with the month number): Investigated date
    category (str): Category to test MAE on. Can be "cold_start", "partially_new" or "normal" (default)
    Returns
    -------
    df_result (dataframe): Initial sales dataframe with the predicted score
    '''
   
    print(f"Calculating MAE for {category}.\n")

    try:
        sales = pd.read_csv(os.path.join(
            directory_path, f'sales_2022-0{selected_date}.csv'), sep=",")
        sales = sales.dropna()
        stores = pd.read_csv(os.path.join(
            directory_path, 'stores.csv'), sep=',')
        products = pd.read_csv(os.path.join(
            directory_path, 'products.csv'), sep=';')

    except FileNotFoundError:
        raise FileNotFoundError(
            f'The files could not be found at {directory_path}. Executing from {os.getcwd()}')

    # Prepare datasets
    sales = sales.merge(products, on=["Material_ID"], how="left")
    sales = sales.merge(stores, on=["Store_Number"], how="left")
    df_unique_stores = stores[["Store_Number"]
                              ].drop_duplicates().reset_index(drop=True)
    df_unique_products = products[["Material_ID"]
                                  ].drop_duplicates().reset_index(drop=True)
    df_all_stores_all_products = df_unique_stores.merge(
        df_unique_products, how='cross')
    df_all_stores_all_products = df_all_stores_all_products.merge(
        sales, on=["Store_Number", "Material_ID"], how="left")
    df_user_item_matrix = df_all_stores_all_products.pivot(
        index="Store_Number", columns="Material_ID", values="Defined_score")
    num_unique_features = sales['Material_ID'].nunique()
   
    print("Number of unique features:", num_unique_features)
    
    # Separation of categories
    if category == 'cold_start':
        if selected_date == 6:
            nbre = 500
        else:
            nbre = 200
        x = df_user_item_matrix.columns[(
            df_user_item_matrix.isna().sum() <= nbre)].tolist()
        half_size = len(x) // 2
        x = random.sample(x, half_size)
        y = df_user_item_matrix[x]
        new = df_user_item_matrix[~df_user_item_matrix.isin(x)]
    elif category == 'partially_new':
        if selected_date == 6:
            nbre = 500
        else:
            nbre = 200
        x = df_user_item_matrix.columns[(
            df_user_item_matrix.isna().sum() <= nbre)].tolist()
        half_size = len(x) // 2
        x = random.sample(x, half_size)
        y = df_user_item_matrix[x]
        new = df_user_item_matrix
        not_null_legit = {}
        for i in x:
            not_null_rows = y.index[pd.notna(y[i])].tolist()
            j = random.sample(not_null_rows, len(not_null_rows)//2)
            for store in j:
                not_null_legit.setdefault(
                    store, []).append((i, y.loc[store][i]))
            new.loc[j, i] = np.nan
    else:
        new = df_user_item_matrix

    matrix = coo_matrix(new.fillna(0).values)

    train_data, test_data, train_row, test_row, train_col, test_col = train_test_split(
        matrix.data, matrix.row, matrix.col, test_size=0.2, random_state=13543)
    train_coo = coo_matrix(
        (train_data, (train_row, train_col)), shape=matrix.shape)
    test_coo = coo_matrix((test_data, (test_row, test_col)),
                          shape=matrix.shape).tocoo()

    train = train_coo.tocsr()
    test = test_coo.tocsr()

    param_combinations = [
        {'no_components': num_unique_features, 'loss': 'warp',
         'item_alpha': 0.0001, 'user_alpha': 0.0001},
        {'no_components': num_unique_features, 'loss': 'warp',
         'item_alpha': 0.0001, 'user_alpha': 0.0001},
        {'no_components': num_unique_features, 'loss': 'warp',
         'item_alpha': 0.0001, 'user_alpha': 0.0001}
    ]

    scores = []
    parameters = []

    # Iterate over parameter combinations for optimisation
    for params in param_combinations:
        model = FMScoreWrapper(**params)
        model.fit(train)
        true_ratings = test.data
        pred_ratings = model.predict(test_coo.row, test_coo.col)
        pred_ratings = np.clip(pred_ratings, 0, 1)
        score = mean_squared_error(true_ratings, pred_ratings)
        scores.append(score)
        parameters.append(params)

    best_idx = np.argmin(scores)
    best_params = parameters[best_idx]

    print('Best parameters:', best_params,'\n')

    best_model = FMScoreWrapper(**best_params)
    best_model.fit(train)
    test_coo = coo_matrix(test)
    true_ratings = test_coo.data
    pred_ratings = best_model.predict(test_coo.row, test_coo.col)
    pred_ratings = np.clip(pred_ratings, 0, 1)

    # Compute MAE in train and test sets
    train_mae = mean_absolute_error(
        train.data, best_model.predict(train_coo.row, train_coo.col))
    test_mae = mean_absolute_error(true_ratings, pred_ratings)

    print('Train Set - MAE:', train_mae)
    print('Test Set - MAE:', test_mae)
    print("User feature matrix dimensions:", matrix.shape,'\n')

    # Get all combinations of stores and products scores
    if category == 'cold_start':
        pred, deja = [], []
        for store in y.index.tolist():
            for product in y.columns.tolist():
                if not (pd.isna(y.loc[store, product])):
                    row = np.array(
                        [df_unique_stores[df_unique_stores['Store_Number'] == store].index[0]])
                    col = np.array(
                        [df_unique_products[df_unique_products['Material_ID'] == product].index[0]])

                    # Predict and transform predictions
                    score = model.predict(row, col)                    
                    score = expit(score)
                    score = np.clip(score, 0, 1)
                    pred += [score]
                    deja += [y.loc[store, product]]

        rmse_sim = np.sqrt(mean_squared_error(pred, deja))
        mae_sim = mean_absolute_error(pred, deja)
        print(f'RMSE Cold Start month {selected_date}: ', rmse_sim)
        print(f'MAE Cold Start month {selected_date}: ', mae_sim)
    elif category == 'partially_new':
        pred, deja = [], []
        for store in list(not_null_legit.keys()):
            for product, sc in not_null_legit[store]:

                row = np.array(
                    [df_unique_stores[df_unique_stores['Store_Number'] == store].index[0]])
                col = np.array(
                    [df_unique_products[df_unique_products['Material_ID'] == product].index[0]])

                # Predict and transform prediction
                score = model.predict(row, col)
                score = expit(score)
                score = np.clip(score, 0, 1)
                pred += [score]
                deja += [sc]

        rmse_sim = np.sqrt(mean_squared_error(pred, deja))
        mae_sim = mean_absolute_error(pred, deja)
        print(f'RMSE Partially New month {selected_date}: ', rmse_sim)
        print(f'MAE Partially New month {selected_date}: ', mae_sim)

# To obtain RMSE and MAE for month 2 for a cold start
model_simulation('data','2','cold_start')

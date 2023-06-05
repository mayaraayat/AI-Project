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


def model_simulation(file_path, selected_date, category):

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

        # Merge sales with stores and products
    sales = sales.merge(products, on=["Material_ID"], how="left")
    sales = sales.merge(stores, on=["Store_Number"], how="left")
    df_unique_stores = stores[["Store_Number"]
                              ].drop_duplicates().reset_index(drop=True)

    # Get unique Products
    df_unique_products = products[["Material_ID"]
                                  ].drop_duplicates().reset_index(drop=True)
    # Define the cross join
    df_all_stores_all_products = df_unique_stores.merge(
        df_unique_products, how='cross')

    # Add score to dataframe
    df_all_stores_all_products = df_all_stores_all_products.merge(
        sales, on=["Store_Number", "Material_ID"], how="left")

    # Define the pivot matrix
    df_user_item_matrix = df_all_stores_all_products.pivot(
        index="Store_Number", columns="Material_ID", values="Defined_score")
    print(df_user_item_matrix)
    num_unique_features = sales['Material_ID'].nunique()
    print("Number of unique features:", num_unique_features)
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

    # Convert the pivot matrix to a sparse matrix
    matrix = coo_matrix(new.fillna(0).values)

    train_data, test_data, train_row, test_row, train_col, test_col = train_test_split(
        matrix.data, matrix.row, matrix.col, test_size=0.2, random_state=13543)
    train_coo = coo_matrix(
        (train_data, (train_row, train_col)), shape=matrix.shape)
    test_coo = coo_matrix((test_data, (test_row, test_col)),
                          shape=matrix.shape).tocoo()

    train = train_coo.tocsr()
    test = test_coo.tocsr()

    # Define the parameter combinations to try
    param_combinations = [
        {'no_components': num_unique_features, 'loss': 'warp',
         'item_alpha': 0.0001, 'user_alpha': 0.0001},
        {'no_components': num_unique_features, 'loss': 'warp',
         'item_alpha': 0.0001, 'user_alpha': 0.0001},
        {'no_components': num_unique_features, 'loss': 'warp',
         'item_alpha': 0.0001, 'user_alpha': 0.0001}
    ]

    # Initialize lists to store the scores and parameter combinations
    scores = []
    parameters = []

    # Iterate over parameter combinations
    for params in param_combinations:
        # Initialize the model with the current parameter combination
        model = FMScoreWrapper(**params)

        # Train the model
        model.fit(train)

        # Get the true ratings from the test set
        true_ratings = test.data

        # Get the corresponding predicted ratings
        pred_ratings = model.predict(test_coo.row, test_coo.col)

        # Clip predictions between 0 and 1
        pred_ratings = np.clip(pred_ratings, 0, 1)

        # Compute the mean squared error
        score = mean_squared_error(true_ratings, pred_ratings)

        # Append the score and parameters to the lists
        scores.append(score)
        parameters.append(params)

        # Find the best parameter combination based on the scores
    best_idx = np.argmin(scores)
    best_params = parameters[best_idx]
    best_score = scores[best_idx]

    print('Best parameters:', best_params)
    print('Best score:', best_score)

    # Train the model with the best parameter combination
    best_model = FMScoreWrapper(**best_params)
    best_model.fit(train)

    # Convert test data to sparse matrix
    test_coo = coo_matrix(test)

    # Get the true ratings from the test set
    true_ratings = test_coo.data

    # Get the corresponding predicted ratings
    pred_ratings = best_model.predict(test_coo.row, test_coo.col)

    # Clip predictions between 0 and 1
    pred_ratings = np.clip(pred_ratings, 0, 1)

    # Compute MAE in train and test sets
    train_mae = mean_absolute_error(
        train.data, best_model.predict(train_coo.row, train_coo.col))
    test_mae = mean_absolute_error(true_ratings, pred_ratings)

    print('Train Set - MAE:', train_mae)
    print('Test Set - MAE:', test_mae)
    print("User feature matrix dimensions:", matrix.shape)

    # Get all combinations of stores and products
    if category == 'cold_start':
        pred, deja = [], []
        for store in y.index.tolist():
            for product in y.columns.tolist():
                if not (pd.isna(y.loc[store, product])):
                    row = np.array(
                        [df_unique_stores[df_unique_stores['Store_Number'] == store].index[0]])
                    col = np.array(
                        [df_unique_products[df_unique_products['Material_ID'] == product].index[0]])

                    # Use the predict method of the LightFM model to get the predicted score
                    score = model.predict(row, col)
                    # Transform predicted scores using sigmoid function
                    score = expit(score)

                    # Clip predictions to (0, 1) range
                    score = np.clip(score, 0, 1)
                    pred += [score]
                    deja += [y.loc[store, product]]
        print(len(pred), len(deja))
        rmse_sim = np.sqrt(mean_squared_error(pred, deja))
        mae_sim = mean_absolute_error(pred, deja)
        print('RMSE cold{}: '.format(selected_date), rmse_sim)
        print('MAE cold: ', mae_sim)
    elif category == 'partially_new':
        pred, deja = [], []
        for store in list(not_null_legit.keys()):
            for product, sc in not_null_legit[store]:

                row = np.array(
                    [df_unique_stores[df_unique_stores['Store_Number'] == store].index[0]])
                col = np.array(
                    [df_unique_products[df_unique_products['Material_ID'] == product].index[0]])

                # Use the predict method of the LightFM model to get the predicted score
                score = model.predict(row, col)
                # Transform predicted scores using sigmoid function
                score = expit(score)

                # Clip predictions to (0, 1) range
                score = np.clip(score, 0, 1)
                pred += [score]
                deja += [sc]
        print(len(pred), len(deja))
        rmse_sim = np.sqrt(mean_squared_error(pred, deja))
        mae_sim = mean_absolute_error(pred, deja)
        print('RMSE pn{}: '.format(date), rmse_sim)
        print('MAE pn: ', mae_sim)

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightfm import LightFM
from scipy.sparse import coo_matrix
from scipy.special import expit
import os


class FMScoreWrapper(LightFM):
    def score(self, X, y=None):
        if y is not None:
            preds = self.predict(X.row, X.col)
            # Negative MSE for optimization
            return -mean_squared_error(y, preds)
        else:
            return 0  # Return 0 when y is None


def recommendation_algorithm(directory_path, selected_date, show_all=True):
    '''
    Parameters
    ----------
    directory_path (str): Path of the directory of files with the data. FMs will be outputted there.
    selected_date (int with the month number): Investigated date
    Returns
    -------
    df_result (dataframe): Initial sales dataframe with the predicted score
    '''
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

    # Convert the pivot matrix to a sparse matrix
    matrix = coo_matrix(df_user_item_matrix.fillna(0).values)

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
    for store_id in df_user_item_matrix.index.tolist():
        pred = []
        for product_id in df_user_item_matrix.columns.tolist():
            if pd.isna(df_user_item_matrix.at[store_id, product_id]):
                # Create a numpy array with the corresponding row and column indices
                row = np.array(
                    [df_unique_stores[df_unique_stores['Store_Number'] == store_id].index[0]])
                col = np.array(
                    [df_unique_products[df_unique_products['Material_ID'] == product_id].index[0]])

            # Use the predict method of the LightFM model to get the predicted score
                score = best_model.predict(row, col)
                score = expit(score)
                score = np.clip(score, 0, 1)
                pred += [(score[0], product_id)]
                df_user_item_matrix.at[store_id, product_id] = score[0]
        if show_all:
            # Top 5 recommendations per store
            print(store_id, sorted(
                pred, key=lambda x: x[0], reverse=True)[:5])

    return df_user_item_matrix


recommendation_algorithm(
    'data', 6, show_all=True)

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from pyfm import pylibfm
import pandas as pd
import os


class FMScoreWrapper(pylibfm.FM):
    def score(self, X, y):
        preds = self.predict(X)
        return -mean_squared_error(y, preds)  # Negative MSE for optimization


def recommendation_algorithm(directory_path, selected_date, show_all=True):
    '''
    Parameters
    ----------
    directory_path (str): Path of the directory of files with the data. FMs will be outputted there.
    selected_date (str with the following format YYYY-MM): Investigated date
    Returns
    -------
    df_result (dataframe): Initial sales dataframe with the predicted score
    '''
    try: # try loading the different csv
        sales = pd.read_csv(
            directory_path + '/sales_2022-0{}'.format(selected_date) + '.csv', sep=",")
        sales = sales.dropna()
        stores = pd.read_csv(directory_path + '/stores.csv', sep=',')
        products = pd.read_csv(directory_path + '/products.csv', sep=';')
        train, test, y_train, y_test = train_test_split(
            sales[['Store_Number', 'Material_ID', 'Defined_score']], sales['Defined_score'], test_size=0.2)
    except Exception as E:
        raise FileNotFoundError('The files could not be found at ' +
                                directory_path + '. Executing from ' + print(os.getcwd()))

    try: # creading the train and test sets
        train.to_csv(directory_path + '/train.csv', sep=',')
        test.to_csv(directory_path + '/test.csv', sep=',')

        def loadData(filename, path=directory_path):
            data = []
            y = []
            stores = set()
            products = set()
            with open(path + '/' + filename) as f:
                lines = f.readlines()
                for line in lines[1:]:
                    (id, store_number, material_id,
                     defined_score) = line.split(',')
                    data.append({"store": str(store_number),
                                "product": str(material_id)})
                    y.append(float(defined_score))
                    stores.add(store_number)
                    products.add(material_id)
            return (data, np.array(y), stores, products)
        
        (train_data, y_train, train_users, train_items) = loadData("train2405.csv")
        (test_data, y_test, test_users, test_items) = loadData("test2405.csv")
    except Exception as E:
        raise ValueError('Data could not be separated into training and test')

    try:
        v = DictVectorizer()
        X_train = v.fit_transform(train_data)
        X_test = v.transform(test_data)
    except:
        raise ValueError('Data could not be vectorised')

    best_mse = np.inf
    best_fm = None
    
    # Parameter optimisation
    num_factors_list = [10, 15, 20]
    num_iter_list = [10, 15, 20]
    learning_rate_list = [0.001, 0.01, 0.1]

    for num_factors in num_factors_list:
        for num_iter in num_iter_list:
            for learning_rate in learning_rate_list:
                fm = FMScoreWrapper(num_factors=num_factors, num_iter=num_iter, verbose=True,
                                    task="regression", initial_learning_rate=learning_rate, learning_rate_schedule="optimal")
                fm.fit(X_train, y_train)
                preds = fm.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                if mse < best_mse:
                    best_mse = mse
                    best_fm = fm

    if best_fm is None:
        raise ValueError("Factorization Machines model could not be created")

    # Predict
    preds = best_fm.predict(X_test)
    mse, mae = mean_squared_error(
        y_test, preds), mean_absolute_error(y_test, preds)
    train_preds = best_fm.predict(X_train)
    train_mse, train_mae = mean_squared_error(
        y_train, train_preds), mean_absolute_error(y_train, train_preds)
    print('Train Set - MAE: ' + str(train_mae) +
          '\nTrain Set - MSE: ' + str(train_mse))

    print('Test Set - MAE: ' + str(mae) + '\nTest Set - MSE: ' + str(mse))

    stores_t = pd.DataFrame(stores['Store_Number'])
    products_t = pd.DataFrame(products['Material_ID'])
    complete_df = stores_t.merge(products_t, how='cross')
    complete_df = complete_df.merge(
        sales, on=["Store_Number", "Material_ID"], how="left")
    df = complete_df.pivot(index="Store_Number",
                           columns="Material_ID", values="Defined_score")

    # Complete the df with the predicted values
    for row in stores['Store_Number']:
        l = []
        for product in products['Material_ID']:
            train_mat_data = [
                {'store': 'Store_Number' + str(row), 'product': product}]
            train_mat = v.transform(train_mat_data)
            if pd.isna(df.at[row, product]):
                df.at[row, product] = best_fm.predict(train_mat)[0]
            l += [best_fm.predict(train_mat)[0]]
        if show_all == True:
            print('Store_Number' + str(row),sorted(l, reverse=True)[:5])
    return df


print(recommendation_algorithm(
    'data', 2, show_all=False))

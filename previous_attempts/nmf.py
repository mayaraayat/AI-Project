import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import NMF
from surprise.model_selection import KFold
from surprise import accuracy


def nmf_pred(file_path, date):
    '''
    Input: 
    - file_path : leads to the csv files with the data
    - date : investigated month
    
    Output:
    - prints the MAE/RMSE for 5 consecutive folds
    '''
    # Load the datasets
    stores = pd.read_csv(file_path+'/stores.csv', sep=",")
    stores = stores.dropna()

    products = pd.read_csv(file_path+'/products.csv', sep=";")
    products = products.dropna()

    sales = pd.read_csv(file_path+"/sales_2022-0{}.csv".format(date))
    sales = sales.dropna()
    sales = sales.drop(columns=["Date", "Sold_Quantity", "Turnover"])

    # Transform dataframes and obtain the essential store/product table
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

    
    mean = df_all_stores_all_products['Defined_score'].mean()
    print('Mean score for month {}:'.format(date), mean)

    
    reader = Reader(rating_scale=(0, 1))
    
    data = Dataset.load_from_df(df_all_stores_all_products[[
        'Store_Number', 'Material_ID', 'Defined_score']].dropna(), reader)
    model = NMF()
    kf = KFold(n_splits=5)
    iteration = 0
    for trainset, testset in kf.split(data):
        model.fit(trainset)
        predictions = model.test(testset)
        print(f'\nMAE for test set {iteration} :', "%.2f" % accuracy.mae(predictions, verbose=True), ', or, as a percentage of mean:',
              accuracy.mae(predictions, verbose=True) / mean)
        print(f'RMSE for the test set {iteration}:', "%.2f" % accuracy.rmse(predictions, verbose=True), 'or, as a percentage of mean:',
              accuracy.rmse(predictions, verbose=True) / mean)
        iteration +=1


nmf_pred('data', 2)

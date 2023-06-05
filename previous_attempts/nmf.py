import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import NMF
from surprise.model_selection import KFold
from surprise import accuracy


def nmf_pred(file_path, date):
    # Load the datasets
    stores = pd.read_csv(file_path+'/stores.csv', sep=",")
    stores = stores.dropna()

    products = pd.read_csv(file_path+'/products.csv', sep=";")
    products = products.dropna()

    sales = pd.read_csv(file_path+"/sales_2022-0{}.csv".format(date))
    sales = sales.dropna()
    sales = sales.drop(columns=["Date", "Sold_Quantity", "Turnover"])

    # Add Products features to the Sales dataframe
    sales = sales.merge(products, on=["Material_ID"], how="left")
    # Add Stores features to the Sales dataframe
    sales = sales.merge(stores, on=["Store_Number"], how="left")

    # Generate the Pivot matrix
    # Get unique Stores
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
    # df_all_stores_all_products['Defined_score'].fillna(df_all_stores_all_products['Defined_score'].mean(), inplace=True)
    # # Define the pivot matrix
    df_user_item_matrix = df_all_stores_all_products.pivot(
        index="Store_Number", columns="Material_ID", values="Defined_score")
    mean = df_all_stores_all_products['Defined_score'].mean()
    print('mean{}'.format(date), mean)

    # Define the reader object
    reader = Reader(rating_scale=(0, 1))
    # Load the data into Surprise's data format
    data = Dataset.load_from_df(df_all_stores_all_products[[
        'Store_Number', 'Material_ID', 'Defined_score']].dropna(), reader)
    model = NMF()
    kf = KFold(n_splits=5)
    for trainset, testset in kf.split(data):
        model.fit(trainset)
        predictions = model.test(testset)
        print(accuracy.mae(predictions, verbose=True),
              accuracy.mae(predictions, verbose=True) / mean)
        print(accuracy.rmse(predictions, verbose=True),
              accuracy.rmse(predictions, verbose=True) / mean)


nmf_pred('data', 2)

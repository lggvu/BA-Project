import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

def stuff(x):
    try:
        y=int(x[-2:])
    except:
        y=int(x[-1:])
    return y
def predict(df):
    df["month"]=df["month_year"].apply(lambda x: x.split("-",1)[1])
    df["id"]=df["product_id"].apply(stuff)
    df_encode =df[["id"]]
    enc = joblib.load('encoder_product_id')
    df_encode = pd.DataFrame(data=enc.transform(df_encode).toarray(), columns=enc.get_feature_names_out(['id']), dtype=bool)
    # trasfer true and false to 1 and 0
    df_encode = df_encode * 1
    df = pd.concat([df, df_encode], axis=1)
    
    df_encode = df[["product_category_name"]]
    enc = joblib.load('encoder_product_category')
    df_encode = pd.DataFrame(data=enc.transform(df_encode).toarray(), columns=enc.get_feature_names_out(['product_category_name']), dtype=bool)
    # trasfer true and false to 1 and 0
    df_encode = df_encode * 1
    df = pd.concat([df, df_encode], axis=1)
    
    df["month"]=df["month"].apply(lambda x: int(x.split("-")[0]))
    def get_month_order(x,y):
        return x+(y-2017)*12
    df["month_order"]=df.apply(lambda x: get_month_order(x['month'], x['year']),axis=1)
    df["total_price_log"]=df["total_price"].apply(lambda x: np.log(x))

    df=df[['freight_price', 'unit_price', 'product_name_lenght',
       'product_description_lenght', 'product_photos_qty', 'product_weight_g',
       'product_score', 'customers', 'weekday', 'weekend', 'holiday', 'month',
       'year', 's', 'volume', 'comp_1', 'fp1', 'comp_2', 'fp2', 'comp_3',
       'fp3', 'lag_price', 'id', 'id_1', 'id_2', 'id_3', 'id_4', 'id_5',
       'id_6', 'id_7', 'id_8', 'id_9', 'id_10',
       'product_category_name_bed_bath_table',
       'product_category_name_computers_accessories',
       'product_category_name_consoles_games',
       'product_category_name_cool_stuff',
       'product_category_name_furniture_decor',
       'product_category_name_garden_tools',
       'product_category_name_health_beauty',
       'product_category_name_perfumery',
       'product_category_name_watches_gifts', 'month_order',"total_price","total_price_log"]]
    float_columns = ['total_price', 'freight_price', 'unit_price', 'product_score', 's',
       'comp_1', 'fp1', 'comp_2', 'fp2', 'comp_3', 'fp3', 'lag_price',
       'total_price_log']
    scaler = joblib.load("scaler.save") 
    df[float_columns] = scaler.transform(df[float_columns])
    df=df.drop(["total_price_log","total_price"],axis=1)
    rf = pickle.load(open("finalized_model.sav", 'rb'))
    y_pred = rf.predict(df)

    dis=6.40982769131368
    m=2.990719731730447
    y_pred=y_pred*dis+m
    y_pred=np.exp(y_pred)

    return y_pred[0]

if __name__ == "__main__": 
    df_test = pd.read_csv("/Users/lggvu/Programming/Business-Analysis/eda/retail_price.csv")
    print(df_test.head(1))
    df_test["product_category_name"].iloc[0] = "burh"
    print(df_test.head(1))
    # print(df_test)
    # print(final(df_test[0:1]))
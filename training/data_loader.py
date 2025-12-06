
import pandas as pd


def load_data():
    try:
        df = pd.read_csv("https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv", index_col=0)
    except Exception as e:
        print("   Could not load data from URL, loading local copy...")
        df = pd.read_csv("../data/fraudTest.csv", index_col=0)  

    df['current_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['trans_date_trans_time', 'unix_time'])
    # for col in ['file', 'date_part']:
    #     df[col] = pd.Series()

    # Convert int to float to avoid issues in ColumnTransformer 
    df = df.astype({col: "float64" for col in df.select_dtypes("int").columns})
    return df
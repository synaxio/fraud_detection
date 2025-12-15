from dotenv import load_dotenv
load_dotenv()

import os
import boto3
from sqlalchemy import create_engine
import pandas as pd

class S3Client:
    def __init__(self, bucket_name):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_dataframe_to_json(self, df: pd.DataFrame):
        # Conversion du dataframe en JSON orient "split"
        for _,row in df.iterrows():
            date_part = row['date_part']
            ts_part = row['file'].split('_')[1]
            file_name = f"transactions/processed/{date_part}/{ts_part}_transaction_processed.json"
            
            json_data = row.to_json(orient="split")

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=json_data,
                ContentType='application/json'
            )
            print(f"Predictions sauvegardées avec succès sur S3") # : s3://{self.bucket_name}/{file_name}")
            
        except Exception as e:
            print(f"Erreur lors de l'enregistrement sur S3: {str(e)}")
  

class DatabaseClient:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)

    def save_dataframe_to_db(self, df, table_name):
        try:
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            print(f"Prédictions ajoutées avec succès sur {self.engine.url.database}.{table_name}")
        except Exception as e:
            print(f"Erreur lors de l'enregistrement en base de données : {str(e)}")

S3 = S3Client(bucket_name=os.getenv("BUCKET_NAME"))
DB = DatabaseClient(connection_string=os.getenv("BACKEND_STORE_URI"))

def save_dataframe_to_s3(df: pd.DataFrame):
    S3.upload_dataframe_to_json(df)

def save_dataframe_to_db(df: pd.DataFrame):
    DB.save_dataframe_to_db(df, table_name="transactions_predicted")
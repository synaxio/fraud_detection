from dotenv import load_dotenv
load_dotenv()

import requests
import pandas as pd
import json
from io import StringIO
import boto3
from datetime import datetime
import os

def _generate_structured_job_id():
    now = datetime.now()
    date_part = now.strftime("%y%m%d")
    ts_part = now.strftime("%H%M%S")
    return date_part, f"{date_part}_{ts_part}"

class APIClient:
    def __init__(self, api_url):
        self.api_url = api_url

    def fetch_data_df(self):
        response = requests.get(self.api_url)
        if response.status_code == 200:
            data = response.json()
            d = json.dumps(json.loads(data))
            df = pd.read_json(d, orient='split')
            return df
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return pd.DataFrame()
        
    def fetch_data_json(self):
        response = requests.get(self.api_url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return {}
        
class S3Client:
    def __init__(self, bucket_name):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_raw_json(self, json_data, file_name):

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=json_data,
                ContentType='application/json'
            )
            #print(f"saved: s3://{self.bucket_name}/{file_name}")
            
        except Exception as e:
            print(f"Error on save: {str(e)}")
    
    def _get_files(self, prefix='transactions/raw/', suffix='_transaction_raw.json'):
        all_objects = []
        continuation_token = None

        while True:
            if continuation_token:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    ContinuationToken=continuation_token
                )
            else:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )

            contents = response.get('Contents', [])
            all_objects.extend([item["Key"] for item in contents if item["Key"].endswith(suffix)])

            # Pagination : vérifier s’il y a d’autres pages
            if response.get('IsTruncated'):  # True si >1000 objets
                continuation_token = response.get('NextContinuationToken')
            else:
                break
            
        print(f"  - Found {len(all_objects)} files with prefix '{prefix}' and suffix '{suffix}' in bucket '{self.bucket_name}'")
        return all_objects

    def get_transactions_dataframe(self, prefix='transactions/raw/', suffix='_transaction_raw.json'):
        files = self._get_files(prefix=prefix, suffix=suffix)
        df= pd.DataFrame(files, columns=['file'])
        df['date_part'] = [s.split('/')[2] for s in df['file']]
        print(f"Downloading {len(df)} files from S3...")
        df['body'] = [S3.s3_client.get_object(Bucket=S3.bucket_name, Key=key)['Body'].read().decode('utf-8') for key in df['file']]
        df = df.reset_index(drop=True)

        df_body = pd.concat([pd.read_json(StringIO(json_str), orient="split") for json_str in df['body']]).reset_index(drop=True)

        df_body = pd.concat([df_body, df[['date_part', 'file']]], axis=1)
        return df_body


S3 = S3Client(bucket_name=os.getenv("BUCKET_NAME"))
API = APIClient(api_url=os.getenv("API_URL"))    

def extract_and_store_raw_transactions():
    date_part, ts_part = _generate_structured_job_id()
    data_json = API.fetch_data_json()
    if data_json!= {}:
        print('New data fetched from API.')
        S3.upload_raw_json(data_json,f"transactions/raw/{date_part}/{ts_part}_transaction_raw.json")
        df =pd.read_json(StringIO(data_json), orient="split")
        df['date_part']= [date_part]
        df['file']= [f"transactions/raw/{date_part}/{ts_part}_transaction_raw.json"]
        
        df = df.astype({col: "float64" for col in df.select_dtypes("int").columns})
        df['current_time'] = df['current_time'].astype(str)
        return df.to_dict(orient='records')
    else:
        print("No data fetched from API.")
        return pd.DataFrame().to_dict(orient='records')
    
def extract_transactions_from_s3(fdate=None):
    if fdate is None:
        transactions = S3.get_transactions_dataframe().reset_index(drop=True)
    else:
        prefix="transactions/raw/" + str(fdate) + '/'
        transactions = S3.get_transactions_dataframe(prefix=prefix)

    transactions = transactions.astype({col: "float64" for col in transactions.select_dtypes("int").columns})
    transactions['current_time'] = transactions['current_time'].astype(str)
    return transactions.to_dict(orient='records')
    

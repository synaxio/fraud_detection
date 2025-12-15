
# %%
# Imports

from dotenv import load_dotenv
load_dotenv()

import requests
import pandas as pd
import json
from io import StringIO
import mlflow

from sqlalchemy import create_engine
import os
import boto3
from datetime import datetime

import time
import sys

import extract
import data_save

# %%
# Fonctions


# %%

if __name__ == "__main__":

    USERNAME = "synaxio"
    API_URL = f"https://{USERNAME}-api-prediction.hf.space"

    headers = {
        "Authorization": f"Bearer {os.getenv('MLFLOW_TRACKING_TOKEN')}"
    }

    # Test boucle complète
    transactions = []
    transaction_id=0
    while True:
        transaction_id += 1
        # if transaction_id >=15: break


        try:
            transactions += extract.extract_and_store_raw_transactions()
        except Exception as e:
            pass

        #break

        if len(transactions) >= 5 :
            print(transactions)
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=transactions,
                    headers={"Content-Type": "application/json",
                            "Authorization": f"Bearer {os.getenv('MLFLOW_TRACKING_TOKEN')}"
                            },
                    timeout=30
                )
                
                print(f"\n✓ Prediction request: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    #print(f"  Prediction: {json.dumps(result, indent=2)}")
                    
                else:
                    print(f"  Error: {response.text}")
                
                    
            except Exception as e:
                print(f"✗ Prediction failed: {e}")

            try:            
                df = pd.concat([pd.DataFrame(transactions), pd.DataFrame(result).rename(columns={'predictions': 'predict_is_fraud'})], axis=1)
                data_save.save_dataframe_to_db(df)
                data_save.save_dataframe_to_s3(df)
                transactions= []
            except Exception as e:
                print(f"✗ save failed: {e}")

            break


        wait_time=12

    # %%
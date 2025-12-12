
from dotenv import load_dotenv
load_dotenv()

import extract
import transform
import data_save

import pandas as pd

import time

if __name__ == "__main__":

    transaction_id=0
    while True:
        transaction_id+=1
        print("Starting ETL cycle: {} ------------------------------------------------------".format(transaction_id))
        
        json_data = extract.extract_and_store_raw_transactions()
        
        if json_data != {}:
            
            df = transform.transform_data(json_data)

        if not df.empty:
            print(f"Transform completed. Uploading processed transaction to S3 and Database...")

            data_save.save_dataframe_to_s3(df)
            data_save.save_dataframe_to_db(df)
        else:
            print('Transform returned an empty DataFrame. Skipping load step.')

        wait_time=12


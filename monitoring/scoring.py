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

from pathlib import Path

local_dir = Path(__file__).resolve().parent

# %%

import mlflow

import shutil
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = 'fraud_detector'

# Récupérer l'expérience
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
exp_id = exp.experiment_id

# Chercher le dernier run réussi
runs = mlflow.search_runs(
    [exp_id], 
    filter_string="attributes.status = 'FINISHED'",        
    order_by=["attributes.start_time DESC"],
    max_results=1
)

last_run_id = runs.iloc[0]["run_id"]
print(f"Last run ID: {last_run_id}")

# Supprimer le dossier précédent
shutil.rmtree("./model", ignore_errors=True)
print("Old model directory removed.")

# Télécharger directement fraud_detector
# On évite list_artifacts qui cause des problèmes avec certains MLflow servers
model_uri = f"runs:/{last_run_id}/fraud_detector"
print(f"Downloading from: {model_uri}")

# %%
try:
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path="./model/"
    )
    print(f"Downloaded to: {local_path}")
    
    # Chercher le dossier 'code' dans la structure téléchargée
    if os.path.exists("fraud_detector/code/MLmodel"):
        print("Found MLmodel in fraud_detector/code/")
        shutil.move("fraud_detector/code", "./model")
        shutil.rmtree("fraud_detector", ignore_errors=True)
        print("Moved fraud_detector/code to model/")
    elif os.path.exists("fraud_detector/MLmodel"):
        print("Found MLmodel in fraud_detector/")
        shutil.move("fraud_detector", "./model")
        print("Moved fraud_detector to model/")
    else:
        # Chercher MLmodel n'importe où dans fraud_detector
        for root, dirs, files in os.walk("fraud_detector"):
            if "MLmodel" in files:
                print(f"Found MLmodel in {root}")
                shutil.move(root, "./model")
                shutil.rmtree("fraud_detector", ignore_errors=True)
                break
    
except Exception as e:
    print(f"Error: {e}")
    raise

# Vérifier que MLmodel existe
if not os.path.exists("model/MLmodel"):
    print("\n❌ MLmodel file not found!")
    print("Contents of current directory:")
    for item in os.listdir("."):
        print(f"  - {item}")
        if os.path.isdir(item):
            print(f"    Contents of {item}:")
            for root, dirs, files in os.walk(item):
                level = root.replace(item, "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for f in files[:5]:
                    print(f"{subindent}{f}")
    raise Exception("MLmodel file not found in downloaded artifacts!")

print("\n✓ Model downloaded successfully")

# Afficher la structure
print("\n=== Final model structure ===")
for root, dirs, files in os.walk("model"):
    level = root.replace("model", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files[:8]:
        print(f"{subindent}{file}")
    if len(files) > 8:
        print(f"{subindent}... and {len(files) - 8} more files")
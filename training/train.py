from dotenv import load_dotenv
load_dotenv()

import data_loader
import preprocessing

import argparse
import os
import pandas as pd
import numpy as np
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

if __name__ == "__main__":

    # ------- Init -------------------------------------------------------
    print('- Init')

    # Set your variables for your environment
    EXPERIMENT_NAME= os.getenv("EXPERIMENT_NAME")
    MLFLOW_TRACKING_URI= os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_TRACKING_TOKEN= os.getenv("MLFLOW_TRACKING_TOKEN")

    # Set tracking URI 
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    #mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    ### MLFLOW Experiment setup
    # # Set experiment's info 
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    # Time execution
    start_time = time.time()

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False) # We won't log models right away

    # ------- Load Data --------------------------------------------------------
    print('- Load Data')
    df = data_loader.load_data()

    count_class_0, count_class_1 = df['is_fraud'].value_counts()

    df_class_0 = df[df['is_fraud'] == 0]
    df_class_1 = df[df['is_fraud'] == 1]

    # Sous-échantillonnage de la classe 0 (on la réduit au même nombre que la classe 1)
    df_class_0_under = df_class_0.sample(count_class_1, random_state=42)

    # Concaténer les deux classes équilibrées
    df_balanced = pd.concat([df_class_0_under, df_class_1])

    # Mélanger les données pour avoir un ordre aléatoire
    df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


    # ------- Preprocessing --------------------------------------------------------
    print('- Preprocessing Data')
    preprocessor = preprocessing.create_preprocessor()
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)

    #X_processed = preprocessor.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ------- Model Training --------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", default=1)
    parser.add_argument("--max_depth", default=None)
    parser.add_argument("--min_samples_split", default=2)
    parser.add_argument("--min_sample_leaf", default=2)
    parser.add_argument("--class_weight", default={0: 1, 1: 1})
    args = parser.parse_args([])

    print('- Model hyperparams')   
    classifier = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        max_depth=None if args.max_depth is None else int(args.max_depth),
        min_samples_split=int(args.min_samples_split),
        min_samples_leaf=int(args.min_sample_leaf),
        class_weight=args.class_weight,
        random_state=42
    )    

    print('- Model Pipeline')
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
        ])

    print('- Log experiment to MLFlow')
    with mlflow.start_run(experiment_id = experiment.experiment_id) as run:
        print('   - fit')
        model.fit(X_train, y_train)
        print('   - predict')
        predictions = model.predict(X_train)

        print('   - score')
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Log model seperately to have more flexibility on setup 
        print('   - log model')
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=EXPERIMENT_NAME,
            registered_model_name="RandomForestClassifier_FraudDetection",
            signature=infer_signature(X_train, predictions),
            code_paths=["training/preprocessing.py"]
        )
    
    




    
    
    

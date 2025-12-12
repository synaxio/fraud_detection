from dotenv import load_dotenv
load_dotenv()

import data_loader
import preprocessing

from data_loader import load_data
from preprocessing import create_preprocessing_pipeline, create_preprocessor

import argparse
import os
import pandas as pd
import numpy as np
import time
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelDef:
    def __init__(self, classifier=None, name='model',grid_params={}, grid_search=False):
        self.classifier = classifier
        self.name = name
        self.grid_params = grid_params
        self.grid_search = grid_search
    

# Définition des configurations de modèles
model_configs = {
        # 'Logistic Regression': ModelDef(classifier=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        #                                 name='Logistic Regression',
        #                                 grid_params= None,
        #                                 grid_search= False
        #                                 ),
        # 'Logistic Regression (GridSearch)': ModelDef(classifier=LogisticRegression(random_state=42),
        #                                 name='Logistic Regression (GridSearch)',
        #                                 grid_params= {
        #                                     'C': [0.1, 1, 10],
        #                                     'penalty': ['l1', 'l2'],
        #                                     'solver': ['liblinear'],
        #                                     'max_iter': [1000],
        #                                     'class_weight': [{0:1, 1:1}, {0:5, 1:5}, {0:0.1, 1:10}]
        #                                     },
        #                                 grid_search= True
        #                                 ),
        'Random Forest' : ModelDef(RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
                                   name= 'Random Forest',
                                   grid_params= None,
                                   grid_search= False
                                   ),
        # 'Random Forest (GridSearch)': ModelDef(classifier= RandomForestClassifier(random_state=42),
        #                                        name= 'Random Forest (GridSearch)',
        #                                        grid_params= {
        #                                            'n_estimators': [2,50,100, 200],
        #                                            'max_depth': [10, 20, None],
        #                                            'min_samples_split': [2, 5],
        #                                            'class_weight': [{0:1, 1:1}, {0:5, 1:5}, {0:0.1, 1:10}]
        #                                            },
        #                                         grid_search= True
        #                                         ),

        'XGBoost': ModelDef(classifier= XGBClassifier(class_weight='balanced', n_estimators=100, objective='binary:logistic', random_state=42),
                            name= 'XGBoost',
                            grid_params= None,
                            grid_search= False),
        
        # 'XGBoost (GridSearch)': ModelDef(classifier= XGBClassifier(n_estimators=100, objective='binary:logistic', random_state=42),
        #                                  name= 'XGBoost (GridSearch)',
        #                                  grid_params= {
        #                                      'max_depth': [3, 5, 7],
        #                                      'min_child_weight': [1, 3, 5],
        #                                      'subsample': [0.6, 0.8, 1.0],
        #                                      'colsample_bytree': [0.6, 0.8, 1.0],
        #                                      'learning_rate': [0.01, 0.1, 0.3],
        #                                      'class_weight': [{0:1, 1:1}, {0:5, 1:5}, {0:0.1, 1:10}]
        #                                      },
        #                                 grid_search= True
        #                                 )
        }


if __name__ == "__main__":

    # ------- Init -------------------------------------------------------
    print('- Init')
    start_time = time.time()

    EXPERIMENT_NAME= os.getenv("EXPERIMENT_NAME")
    MLFLOW_TRACKING_URI= os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_TRACKING_TOKEN= os.getenv("MLFLOW_TRACKING_TOKEN")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    model_definition = model_configs['Random Forest']

    mlflow.sklearn.autolog(log_models=False) # We won't log models right away

    # ------- Load Data --------------------------------------------------------
    print('- Load Data')
    df = data_loader.load_data()

    # ------- Preprocessing --------------------------------------------------------
    print('- Preprocessing Data')
    #preprocessor = create_preprocessing_pipeline()
    preprocessor = create_preprocessor()

    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


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

    #classifier = model_definition.classifier

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


    # # ✅ Créer UNE SEULE run avec tout le contrôle
    # with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_definition.name) as run:
        
    #     # ✅ Activer autolog DANS le contexte de la run
    #     mlflow.sklearn.autolog(
    #         log_models=False,  # On va logger manuellement pour avoir le contrôle
    #         log_input_examples=False,
    #         log_model_signatures=False
    #     )
        
    #     print('- Model Training')
    #     model = Pipeline(steps=[
    #         ('preprocessor', preprocessor),
    #         ('classifier', model_definition.classifier)
    #     ])
        
    #     # Le fit() loggera automatiquement les params et metrics
    #     model.fit(X_train, y_train)

    #     print('- Predict & Evaluate')
    #     predictions = model.predict(X_test)
    #     test_score = model.score(X_test, y_test)
        
    #     # Logs additionnels si besoin
    #     mlflow.log_metric("test_score", test_score)
    #     mlflow.log_param("test_size", 0.2)

    #     print('- Log model avec tous les paramètres personnalisés')
    #     mlflow.sklearn.log_model(
    #         sk_model=model,
    #         artifact_path=EXPERIMENT_NAME,  # ✅ Nom de l'artifact path
    #         registered_model_name=model_definition.name,  # ✅ Nom dans le Registry
    #         signature=infer_signature(X_train, predictions),  # ✅ Signature
    #         code_paths=["training/preprocessing.py"],  # ✅ Votre code sauvegardé
    #         input_example=X_train.iloc[:5]  # ✅ Exemple d'input (optionnel)
    #     )
        
    #     # ✅ Logger des fichiers additionnels si besoin
    #     mlflow.log_artifact("training/preprocessing.py", artifact_path="code")
        
    #     print(f"✅ Run ID: {run.info.run_id}")
    #     print(f"✅ Model registered as: {model_definition.name}")
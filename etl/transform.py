from dotenv import load_dotenv
load_dotenv()

import mlflow
import pandas as pd
from io import StringIO
import os


def _load_last_model(experiment_name, mlflow_tracking_uri):

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    if not experiment_name:
        raise ValueError("La variable d'environnement EXPERIMENT_NAME n'est pas définie.")
    else:
        print(f"Utilisation de l'expérience MLflow : {experiment_name}")

    # Récupération de l'ID de l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Expérience MLflow '{experiment_name}' introuvable.")

    experiment_id = experiment.experiment_id

    # Récupération du dernier run terminé
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],  # du plus récent au plus ancien
        max_results=1
    )

    if runs.empty:
        raise ValueError(f"Aucun run terminé trouvé pour l'expérience '{experiment_name}'.")

    last_run_id = runs.iloc[0].run_id
    
    # Construction du chemin du modèle dans MLflow
    model_name = experiment_name  # adapte si besoin
    logged_model = f"runs:/{last_run_id}/{model_name}"

    # Chargement du modèle
    print(f"Chargement du modèle depuis MLflow : {logged_model}")
    return mlflow.pyfunc.load_model(logged_model)
    
# Récupération du modèle MLflow
model = _load_last_model(experiment_name=os.getenv("EXPERIMENT_NAME"), mlflow_tracking_uri= os.getenv("MLFLOW_TRACKING_URI"))    

def transform_data(json_data ) -> pd.DataFrame:
    if len(json_data) > 0:
        
        #df =pd.read_json(StringIO(json_data))
        df = pd.DataFrame(json_data)
        df = df.astype({col: "float64" for col in df.select_dtypes("int").columns})
        col_exclude = ['is_fraud', 'file', 'date_part']
        df_exclude = df[col_exclude]
        df.drop(columns=col_exclude, inplace=True)  

        df['predict_is_fraud'] = model.predict(df)

        df = pd.concat([df, df_exclude], axis=1)
        
        return df
    
    else:
        print("No data provided for prediction.")
        return pd.DataFrame()
    
    

                       


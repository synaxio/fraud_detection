import mlflow.pyfunc
import os

def load_model(model_path="/app/model"):  # ← Valeur par défaut ajoutée
    """Charge le modèle MLflow depuis le chemin spécifié."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le dossier du modèle '{model_path}' n'existe pas")
    
    # Vérifier que MLmodel existe
    mlmodel_path = os.path.join(model_path, "MLmodel")
    if not os.path.exists(mlmodel_path):
        raise FileNotFoundError(f"Le fichier MLmodel n'existe pas dans '{model_path}'")
    
    model = mlflow.pyfunc.load_model(model_path)
    return model
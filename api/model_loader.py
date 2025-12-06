import mlflow.pyfunc

def load_model():
    model_path = "/app/model"
    return mlflow.pyfunc.load_model(model_path)
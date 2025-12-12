import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score, auc,
                             precision_score, recall_score, precision_recall_curve,
                             )
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer personnalisé pour l'ingénierie de features
    """
    
    def __init__(self):
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        # self.feature_names_ = X.columns
        return self

    
    def transform(self, X):
        
        #X_transformed = X.copy()
        
        X_transformed = X.astype({col: "float64" for col in X.select_dtypes(["int", "int64"]).columns}).copy()

        # Calcul de la distance géographique   
        # def compute_distance(X):
        #     lat1 = X['lat']; lon1 = X['long']
        #     lat2 = X['merch_lat']; lon2 = X['merch_long']
        #     R = 6371000
        #     phi1, phi2 = np.radians(lat1), np.radians(lat2)
        #     d_phi = np.radians(lat2 - lat1)
        #     d_lambda = np.radians(lon2 - lon1)
        #     a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lambda/2)**2
        #     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        #     return c * R / 1000
        

        # Calcul de la distance géographique
        lat1, lon1 = X_transformed['lat'], X_transformed['long']
        lat2, lon2 = X_transformed['merch_lat'], X_transformed['merch_long']
        R = 6371  # km
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        d_phi = np.radians(lat2 - lat1)
        d_lambda = np.radians(lon2 - lon1)
        a = np.sin(d_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        X_transformed['distance_km'] = R * c

        

        # X_transformed['distance_km'] = compute_distance(X_transformed)

        X_transformed["dob"] = pd.to_datetime(X_transformed["dob"], errors="coerce")
        today = pd.Timestamp.today()
        X_transformed["age"] = (today - X_transformed['dob']).dt.days // 365

        X_transformed['cb_len'] = [len(str(int(s))) for s in X_transformed['cc_num']]

        # Extraction temporelle
        X_transformed['current_time'] = pd.to_datetime(X_transformed['current_time'])
        X_transformed['hour'] = X_transformed['current_time'].dt.hour
        X_transformed['day_of_week'] = X_transformed['current_time'].dt.dayofweek.astype("object")
        X_transformed['is_weekend'] = X_transformed['day_of_week'].isin([5, 6]).astype(int).astype("object")
        
        # Encodage cyclique des heures
        X_transformed['hour_sin'] = np.sin(2 * np.pi * X_transformed['hour'] / 24)
        X_transformed['hour_cos'] = np.cos(2 * np.pi * X_transformed['hour'] / 24)
        X_transformed['hour'] = X_transformed['hour'].astype("object")
        
        return X_transformed
    
    # def get_feature_names_out(self, input_features=None):
    #     return self.feature_names_


def create_preprocessing_pipeline():
    """
    Crée un pipeline de preprocessing complet avec feature engineering
    """
    
    # Colonnes après feature engineering
    numeric_cols = [
        "amt", "city_pop", "distance_km",
        "hour_sin", "hour_cos", 'age', 'cb_len'
    ]

    categorical_cols = [
        "gender", "category", 'job', "day_of_week", "is_weekend", 'hour'
    ]
    
    # ✅ ColumnTransformer qui s'applique APRÈS le feature engineering
    preprocess_pipeline = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ],
        remainder="drop"
    )   

    # ✅ Pipeline final : feature engineering PUIS preprocessing
    full_pipeline = Pipeline([
        ("feature_engineering", FeatureEngineeringTransformer()),  # Travaille sur DataFrame
        ("preprocess", preprocess_pipeline)  # Travaille sur le résultat transformé
    ])

    return full_pipeline

    
# def transform_data(df : pd.DataFrame, test_size=0.2, random_state=42, stratify=None):
#     print("Création du pipeline de preprocessing...")
#     pipeline = create_preprocessing_pipeline()

#     # --- Séparation train/test AVANT toute transformation ---
#     X = df.drop(columns=["is_fraud"])
#     y = df["is_fraud"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=test_size,
#         random_state=random_state,
#         stratify=stratify
#     )

#     # Fit sur le TRAIN
#     X_train = pipeline.fit_transform(X_train)

#     # Transform simple sur le TEST
#     X_test = pipeline.transform(X_test)

#     return X_train, X_test, y_train, y_test

# def create_transformers():   
#     return FeatureEngineeringTransformer(), CategoricalEncoderTransformer()
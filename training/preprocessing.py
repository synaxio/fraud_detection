import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

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

    # categorical_cols = [
    #     "gender", "category", 'job', "day_of_week", "is_weekend", 'hour'
    # ]

    categorical_cols = [
         "category" #, "gender", 'job', "day_of_week", "is_weekend", 'hour'
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

class FeatureBuilderAndEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_cols = None
        self.cat_cols = None
        
        self.num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

        self.cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

    def create_features(self, X):

        def compute_distance(X):
            lat1 = X['lat']; lon1 = X['long']
            lat2 = X['merch_lat']; lon2 = X['merch_long']
            R = 6371
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            d_phi = np.radians(lat2 - lat1)
            d_lambda = np.radians(lon2 - lon1)
            a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            #return pd.DataFrame({'distance': c * R})
            return c * R

        X = X.copy()

        X['current_time'] = pd.to_datetime(X['current_time'], errors="coerce")
        X["dob"] = pd.to_datetime(X["dob"], errors="coerce")

        # Conversion des colonnes int en float
        X = X.astype({col: "float64" for col in X.select_dtypes(["int", "int64"]).columns})

        # 🔵 Exemples de colonnes numériques dérivées
        # distance entre lieu de transaction et position du marchand
        X['distance'] = compute_distance(X)
        
        # age du client
        today = pd.Timestamp.today()
        X["age"] = (today - X['dob']).dt.days // 365
        #X['age']= [pd.to_numeric(2025 - pd.to_datetime(X['dob']).dt.year)]
        
        X['cb_len'] = [len(str(int(s))) for s in X['cc_num']]

        # heure de la transaction
        X['transaction_hour'] = X['current_time'].dt.hour

        # 🔴 Exemple de colonne catégorielle dérivée
        #X['author']= X['first'].astype(str) + '_' + X['last'].astype(str) + '_' + X['gender'].astype(str)

        # Extraction temporelle
        X['hour'] = X['current_time'].dt.hour
        X['day_of_week'] = X['current_time'].dt.dayofweek.astype("object")
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int).astype("object")
        
        # Encodage cyclique des heures
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['hour'] = X['hour'].astype("object")

        # Colonnes à dropper après création des features
        drop_cols = ['first', 'last', 'gender', 'cc_num', 'merchant', 'street', 'city', 'state', 'zip', 'city_pop','job', 'trans_num', 'current_time', 'lat', 'long', 'merch_lat', 'merch_long', 'dob'] #, 'file', 'date_part']
        X = X.drop(columns=drop_cols)

        return X

    def fit(self, X, y=None):
        X = self.create_features(X)

        # ⚡ Détection automatique des types
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Fit scaler + encoder
        self.num_pipeline.fit(X[self.num_cols])
        self.cat_pipeline.fit(X[self.cat_cols])

        return self

    def transform(self, X):
        X = self.create_features(X)

        X_num = self.num_pipeline.transform(X[self.num_cols])
        X_cat = self.cat_pipeline.transform(X[self.cat_cols]).toarray()

        num_feature_names = self.num_cols

        # noms des colonnes catégorielles après OneHot
        ohe = self.cat_pipeline.named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(self.cat_cols).tolist()

        # concaténation
        all_cols = num_feature_names + cat_feature_names

        X_final = pd.DataFrame(
            np.hstack([X_num, X_cat]),
            columns=all_cols
        )

        # # Concaténation
        # X_final = pd.DataFrame(np.hstack([X_num, X_cat]), columns=self.num_cols + self.cat_cols)

        return X_final
    
def create_preprocessor():   
    #return basic_preprocessor() 
    return FeatureBuilderAndEncoder()
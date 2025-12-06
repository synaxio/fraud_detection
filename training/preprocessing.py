import pandas as pd
import numpy as np

from sklearn.preprocessing import  StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

def basic_preprocessor():
# Creation d'une colonne author
    def create_author(X):
        return pd.DataFrame({
            'author': X['first'].astype(str) + '_' + X['last'].astype(str) + '_' + X['gender'].astype(str)
        })

    # calcul de distance
    def compute_distance(X):
        lat1 = X['lat']; lon1 = X['long']
        lat2 = X['merch_lat']; lon2 = X['merch_long']
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        d_phi = np.radians(lat2 - lat1)
        d_lambda = np.radians(lon2 - lon1)
        a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return pd.DataFrame({'distance': c * R})

    # ajout de l'age
    def compute_age(X):
        return pd.DataFrame({'age': pd.to_numeric(2025 - pd.to_datetime(X['dob']).dt.year)})
    
    # conversion int en float
    def convert_int_to_float(X):
        for col in X.select_dtypes("int").columns:
            X[col] = X[col].astype("float64")
        return X

    # --- Pipelines ---
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        #('int_to_float', FunctionTransformer(convert_int_to_float, validate=False)),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    distance_pipeline = Pipeline([
        ('distance', FunctionTransformer(compute_distance, validate=False)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    author_pipeline = Pipeline([
        ('author', FunctionTransformer(create_author, validate=False)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    age_pipeline = Pipeline([
        ('age', FunctionTransformer(compute_age, validate=False)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    # --- Colonnes d'origine (hors dérivées) ---
    numeric_features = ['amt']  
    categorical_features = ['merchant', 'category', 'city', 'state', 'zip', 'job'] 

    preprocessor = ColumnTransformer(
        transformers=[
            ('distance', distance_pipeline, ['lat', 'long', 'merch_lat', 'merch_long']),
            ('author', author_pipeline, ['first', 'last', 'gender']),
            #('age', age_pipeline, ['dob']),

            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop',  # drop de tout ce qui n'est pas dans les pipelines
    )

    return preprocessor


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
            R = 6371000
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
        
        # heure de la transaction
        X['transaction_hour'] = X['current_time'].dt.hour

        # 🔴 Exemple de colonne catégorielle dérivée
        #X['author']= X['first'].astype(str) + '_' + X['last'].astype(str) + '_' + X['gender'].astype(str)

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
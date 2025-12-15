

from data_loader import load_data
from preprocessing import create_preprocessing_pipeline, create_preprocessor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score, auc,
                             precision_score, recall_score, precision_recall_curve,
                             )

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FraudDataAnalyse:
    """
    Fonctions d'analyse du dataset d'apprentissage
    """
    
    def __init__(self, df):
        self.df = df.copy()
        # self.balance_method = balance_method
        # self.experiment_name = experiment_name
        # self.pipeline = None
        # self.X_train = None
        # self.X_test = None
        # self.y_train = None
        # self.y_test = None
        
        
    def basic_statistics(self):
        """Affiche les statistiques générales du dataframe"""
        print("="*50)
        print("STATISTIQUES GÉNÉRALES DU DATASET")
        print("="*50)
        
        print(f"Shape du dataset: {self.df.shape}")
        print(f"Taille en mémoire: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n--- Types de données ---")
        print(self.df.dtypes.value_counts())
        
        print("\n--- Valeurs manquantes ---")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Colonne': missing_data.index,
            'Valeurs manquantes': missing_data.values,
            'Pourcentage': missing_percent.values
        })
        print(missing_df[missing_df['Valeurs manquantes'] > 0])
        
        print("\n--- Statistiques descriptives pour les variables numériques ---")
        print(self.df.describe())
    
    def fraud_distribution_viz(self):
        """Visualise la distribution des fraudes"""
        plt.figure(figsize=(12, 6))
        
        # Pie chart
        plt.subplot(1, 2, 1)
        fraud_counts = self.df['is_fraud'].value_counts()
        labels = ['Non-fraude', 'Fraude']
        colors = ["#4440fc", "#e43922"] 
        
        plt.pie(fraud_counts.values, labels=labels, autopct='%1.2f%%', 
                colors=colors, startangle=90, explode=(0.1, 0))
        plt.title('Distribution des Fraudes dans le Dataset')
        
        # Bar chart
        plt.subplot(1, 2, 2)
        fraud_counts.plot(kind='bar', color=colors)
        plt.title('Nombre de Transactions par Type')
        plt.xlabel('Type de Transaction (0=Non-fraude, 1=Fraude)')
        plt.ylabel('Nombre de Transactions')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Pourcentage de fraudes: {(fraud_counts[1] / fraud_counts.sum()) * 100:.2f}%")
    
    def correlation_matrix(self):
        """Affiche la matrice de corrélation"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(15, 12))
        mask = np.triu(corr_matrix)
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                    center=0, square=True, fmt='.2f')
        plt.title('Matrice de Corrélation des Variables Numériques')
        plt.tight_layout()
        plt.show()
        
        fraud_corr = corr_matrix['is_fraud'].abs().sort_values(ascending=False)
        print("\nCorrélations avec is_fraud (valeur absolue):")
        print(fraud_corr.head(10))
    
    def fraud_by_hour_viz(self):
        """Visualise la répartition des fraudes par heure et par jour"""
        # Appliquer temporairement le feature engineering pour la visualisation
        df_viz = self.df.copy()
        df_viz['current_time'] = pd.to_datetime(df_viz['current_time'])
        df_viz['hour'] = df_viz['current_time'].dt.hour
        df_viz['day_of_week'] = df_viz['current_time'].dt.dayofweek
        
        plt.figure(figsize=(15, 8))
        
        # Distribution générale par heure
        plt.subplot(2, 2, 1)
        hour_dist = df_viz['hour'].value_counts().sort_index()
        hour_dist.plot(kind='bar')
        plt.title('Distribution des Transactions par Heure')
        plt.xlabel('Heure du Jour')
        plt.ylabel('Nombre de Transactions')
        
        # Taux de fraude par jour de la semaine
        plt.subplot(2, 2, 2)
        fraud_rate_by_day = df_viz.groupby('day_of_week')['is_fraud'].mean()
        fraud_rate_by_day.plot(kind='bar', color='#e74c3c')
        plt.title('Taux de Fraude par Jour de la Semaine')
        plt.xlabel('Jour de la Semaine (0=Lundi)')
        plt.ylabel('Taux de Fraude')
        plt.xticks(rotation=0)
        
        # Taux de fraude par heure
        plt.subplot(2, 2, 3)
        fraud_rate_by_hour = df_viz.groupby('hour')['is_fraud'].mean()
        fraud_rate_by_hour.plot(kind='line', marker='o', color='orange')
        plt.title('Taux de Fraude par Heure')
        plt.xlabel('Heure du Jour')
        plt.ylabel('Taux de Fraude')
        
        # Heatmap par jour de la semaine et heure
        plt.subplot(2, 2, 4)
        fraud_heatmap = df_viz.groupby(['day_of_week', 'hour'])['is_fraud'].mean().unstack()
        sns.heatmap(fraud_heatmap, cmap='Reds', annot=False)
        plt.title('Taux de Fraude par Jour et Heure')
        plt.ylabel('Jour de la Semaine (0=Lundi)')
        plt.xlabel('Heure du Jour')
        
        plt.tight_layout()
        plt.show()

class FraudModel:
    """
    Classe pour gérer l'entraînement et l'évaluation d'un modèle de détection de fraude
    """
    
    def __init__(self, model, model_name, grid_params=None, use_gridsearch=False):
        """
        Parameters:
        -----------
        model : sklearn estimator
            Instance du modèle à entraîner (ex: LogisticRegression(), RandomForestClassifier())
        model_name : str
            Nom du modèle pour l'affichage
        grid_params : dict
            Grille de paramètres pour GridSearchCV
        use_gridsearch : bool
            Utiliser GridSearch pour l'optimisation
        """
        #self.base_model = model
        self.model_name = model_name
        self.grid_params = grid_params
        self.use_gridsearch = use_gridsearch
        self.base_model = Pipeline(steps=[
                ('preprocessor', create_preprocessor()),
                ('classifier', model)
                ])
        self.results = {}
        
    def train(self, X_train, y_train, X_test, y_test):
        """Entraîne le modèle avec ou sans GridSearch"""
        print(f"\n{'='*60}")
        print(f"Entraînement: {self.model_name}")
        print(f"{'='*60}")
        
        if self.use_gridsearch and self.grid_params:
            print("Optimisation avec GridSearchCV...")
            grid_search = GridSearchCV(
                self.base_model,
                self.grid_params,
                cv=5,
                scoring='average_precision',
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Meilleurs paramètres: {grid_search.best_params_}")
            self.results['best_params'] = grid_search.best_params_
        else:
            print("Entraînement avec paramètres par défaut...")
            self.model = self.base_model
            self.model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calcul des métriques
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Courbe Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall_curve, precision_curve)
        self.results.update({
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'f1_score': f1_score(y_test, y_pred),
            'auc_pr': auc_pr,                     # <-- remplace roc_auc
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve
        })
        
        print(f"\nRésultats:")
        print(f"  F1-Score: {self.results['f1_score']:.4f}")
        print(f"  ROC-PR: {self.results['auc_pr']:.4f}")
        print(f"  Précision: {self.results['precision']:.4f}")
        print(f"  Rappel: {self.results['recall']:.4f}")
        
        return self.results
    
    def visualize_performance(self, figsize=(14, 5)):
        """Visualise la courbe ROC et la matrice de confusion"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Courbe ROC
        ax = axes[0]
        precision, recall, _ = precision_recall_curve(
            self.results['y_test'], 
            self.results['y_pred_proba']
        )
        auc_pr = auc(recall, precision)

        ax.plot(recall, precision, linewidth=2, label=f"AUC-PR = {auc_pr:.3f}")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Courbe Precision-Recall - {self.model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Matrice de confusion
        ax = axes[1]
        sns.heatmap(self.results['confusion_matrix'], annot=True, fmt='d', 
                    cmap='Blues', ax=ax, cbar=True)
        ax.set_title(f'Matrice de Confusion - {self.model_name}')
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Réalité')
        ax.set_xticklabels(['Non-Fraude', 'Fraude'])
        ax.set_yticklabels(['Non-Fraude', 'Fraude'])
        
        plt.tight_layout()
        plt.show()
    
    def get_metrics_dict(self):
        """Retourne un dictionnaire avec toutes les métriques"""
        return {
            'Modèle': self.model_name,
            'F1-Score': self.results['f1_score'],
            'AUC-PR': self.results['auc_pr'],
            'Précision': self.results['precision'],
            'Rappel': self.results['recall'],
            'Spécificité': self.results['specificity'],
            'VP': self.results['tp'],
            'VN': self.results['tn'],
            'FP': self.results['fp'],
            'FN': self.results['fn']
        }


class ModelComparator:
    """Classe pour comparer plusieurs modèles"""
    
    def __init__(self):
        self.models = []
        self.comparison_df = None
    
    def add_model(self, fraud_model):
        """Ajoute un modèle entraîné à la comparaison"""
        self.models.append(fraud_model)
    
    def create_comparison_table(self):
        """Crée un tableau comparatif de tous les modèles"""
        comparison_data = [model.get_metrics_dict() for model in self.models]
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.round(4)
        
        print("\n" + "="*100)
        print("TABLEAU COMPARATIF DES MODÈLES")
        print("="*100)
        print(self.comparison_df.to_string(index=False))
        print("="*100)
        
        return self.comparison_df
    
    def visualize_comparison(self):
        """Visualise la comparaison des modèles"""
        if self.comparison_df is None:
            self.create_comparison_table()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ---- F1-Scores ----
        ax = axes[0, 0]
        bars = ax.barh(self.comparison_df['Modèle'], 
                       self.comparison_df['F1-Score'], 
                       color='steelblue')
        ax.set_xlabel('F1-Score')
        ax.set_title('Comparaison des F1-Scores')
        for i, (bar, score) in enumerate(zip(bars, self.comparison_df['F1-Score'])):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        # ---- AUC-PR  ----
        ax = axes[0, 1]
        bars = ax.barh(self.comparison_df['Modèle'], 
                       self.comparison_df['AUC-PR'], 
                       color='coral')
        ax.set_xlabel('AUC-PR')
        ax.set_title('Comparaison des AUC-PR')
        for i, (bar, score) in enumerate(zip(bars, self.comparison_df['AUC-PR'])):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        # ---- Précision vs Rappel ----
        ax = axes[1, 0]
        ax.scatter(self.comparison_df['Rappel'], 
                   self.comparison_df['Précision'], 
                   s=200, alpha=0.6,
                   c=range(len(self.comparison_df)), cmap='viridis')
        
        for i, model in enumerate(self.comparison_df['Modèle']):
            ax.annotate(
                model,
                (self.comparison_df['Rappel'].iloc[i],
                 self.comparison_df['Précision'].iloc[i]),
                fontsize=9, ha='center'
            )
        
        ax.set_xlabel('Rappel')
        ax.set_ylabel('Précision')
        ax.set_title('Précision vs Rappel')
        ax.grid(True, alpha=0.3)
        
        # ---- Comparaison multi-métriques ----
        ax = axes[1, 1]
        # AUC-PR remplace ROC-AUC dans les métriques affichées
        metrics = ['F1-Score', 'AUC-PR', 'Précision', 'Rappel', 'Spécificité']
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, model_name in enumerate(self.comparison_df['Modèle']):
            values = self.comparison_df[self.comparison_df['Modèle'] == model_name][metrics].values[0]
            ax.bar(x + i*width, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Métriques')
        ax.set_ylabel('Score')
        ax.set_title('Comparaison Multi-Métriques')
        ax.set_xticks(x + width * (len(self.comparison_df) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()





# Définition des configurations de modèles
model_configs = [
        {
            'model': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            'model_name': 'Logistic Regression',
            'grid_params': None,
            'use_gridsearch': False
        },
        {
            'model': LogisticRegression(random_state=42),
            'model_name': 'Logistic Regression (GridSearch)',
            'grid_params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear'],
                'classifier__max_iter': [1000],
                'classifier__class_weight': [{0:1, 1:1}, {0:5, 1:5}, {0:0.1, 1:10}]
            },
            'use_gridsearch': True
        },
        {
            'model': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
            'model_name': 'Random Forest',
            'grid_params': None,
            'use_gridsearch': False
        },
        {
            'model': RandomForestClassifier(random_state=42),
            'model_name': 'Random Forest (GridSearch)',
            'grid_params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10]
                #'classifier__class_weight': [{0:1, 1:1}, {0:5, 1:5}, {0:0.1, 1:10}]
            },
            'use_gridsearch': True
        },
        {
            'model': XGBClassifier(n_estimators=200,  random_state=42, eval_metric='aucpr',  tree_method='hist' ),
            'model_name': 'XGBoost',
            'grid_params': None,
            'use_gridsearch': False
        },
        {
            'model': XGBClassifier(n_estimators=200,  random_state=42, eval_metric='aucpr',  tree_method='hist' ),
            'model_name': 'XGBoost (GridSearch)',
            'grid_params': {
                'classifier__max_depth': [3, 5, 7],
                'classifier__scale_pos_weight': [10, 50, 100],
                'classifier__learning_rate': [0.01, 0.1, 0.3]
            },
            'use_gridsearch': True
        }
    ]

       
#model_configs = {k: v for k, v in model_configs.items() if k in ['Logistic Regression', 'Random Forest', 'SVM']}
#model_configs = [model_configs[i] for i in [4,2,0]]
#model_configs = [model_configs[i] for i in [0,2,4]]
model_configs = [model_configs[i] for i in [2,3,4,5]]

def main():
    
    # Chargement des données
    df = load_data()
    


    # Étape 1: Analyse exploratoire
    print("\n" + "="*60)
    print("PHASE 1: ANALYSE EXPLORATOIRE")
    print("="*60)
    
    experiment = FraudDataAnalyse(df)
    experiment.basic_statistics()
    experiment.fraud_distribution_viz()
    experiment.correlation_matrix()
    experiment.fraud_by_hour_viz()
    
    # Etape 2: Preprocessing
    print('- Preprocessing Data')

    preprocessor = create_preprocessor()

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Étape 3: Configuration et entraînement des modèles
    print("\n" + "="*60)
    print("PHASE 3: ENTRAÎNEMENT DES MODÈLES")
    print("="*60)
     
    # Entraînement de tous les modèles
    comparator = ModelComparator()
    
    for config in model_configs:
        model = FraudModel(**config)
        model.train(X_train, y_train, X_test, y_test)
        model.visualize_performance()
        comparator.add_model(model)
    
    # Étape 4: Comparaison finale
    print("\n" + "="*60)
    print("PHASE 4: COMPARAISON FINALE")
    print("="*60)
    
    comparison_df = comparator.create_comparison_table()
    comparator.visualize_comparison()
    
    print("\n✅ Analyse complète terminée!")
    
    return experiment, comparator, comparison_df


if __name__ == "__main__":
    main()
# JEDHA - Projet final - CI/CD fraud detection

## Enjeux du projet 
Mettre en oeuvre une chaine CI/CD de détection de fraude en temps réel
Ce projet comporte 3 parties:
  - Choix d'un modèle initial
  - Mise en ligne d'une API permettant d'exploiter ce modèle
  - Améliorer le modèle en continu

### I. Choix du modèle
  #### Benchmark et optimisation
  - Récupération des données d'entrainement depuis un bucket S3
  - Préparation des données
  - Benchmark modèles rapide (ici, Regression logistique, Random Forest et XGBoost) 
  - GridSearch et compraison sur les deux meilleurs modèles
    <img width="1026" height="926" alt="image" src="https://github.com/user-attachments/assets/f5f6ac3c-cc3c-4c70-ba40-29b6a4ed88b7" />
  - Stockage du modèle et de ses métrics via un serveur MLFlow hébergé sur HuggingFace, qui <br>
  enregistre les données dans un bucket S3 et une base Postresql (NeonDB)
<img width="1025" height="486" alt="image" src="https://github.com/user-attachments/assets/fa1d2b18-54ec-4b80-807e-c8e4b0e56fd2" />

### II. API
Le modèle entrainé est déployé sur une API hébergée sur HuggingFace. L'API attend un tableau de <br>
transactions, pour lesquelles elle renvoie 0 ou 1 en fonction de la detection d'une fraude (1).
🚨Au moment du push vers GitHub du code d'entrainement, GitAction réalise des tests sur le code et <br>
sur les fonctions principales, ainsi que les données utilisées et générées. Si les tests sont <br>
concluants, le modèle est déployé vers l'API.
Particularité: bien qu'HuggingFace permette de déployer facilement une API dans un docker, des problèmes <br>
de dépendances ont été rencontrés avec des modèles qui ne sont pas exclusivement issus de scikit-learn,<br>
particulièrement XGBoost. <br>
La solution de contournement a été de forcer le rebuild de l'API au moment d'une nouvelle version, <br>
plutôt que simplement "appeler" la nouvelle version depuis MLFLOW, et de lui passer directement la <br>
totalité des fichiers nécessaires. 

### III. CI/CD
#### Principe:
Il y a en permanence un modèle "candidat" en parallèle du modèle déployé sur l'API. Périodiquement, <br>
un scoring des deux modèles est fait sur un jeu de données récentes et labelisées. Le meilleur des <br>
deux modèles est déployé sur l'API, le second est réentrainé sur les données les plus récentes et <br>
devient "candidat", en attente d'un nouveau scoring.
<img width="915" height="753" alt="image" src="https://github.com/user-attachments/assets/397d8db4-9d14-4092-aad4-7380370008cf" />

Une intervention manuelle sur un nouveau modèle a pour conséquence de remplacer le candidat actuel.

### IV. Simulation de la consommation de l'API
   #### ETL
   L'API devrait être consommée chaque fois qu'un topic pousserait une nouvelle transaction. Dans ce projet, le topic est remplacé par un script déployé via un docker local. Ce srcipt récupère en permanence les nouvelles transactions émises et les soumet par paquet à l'API. Tant que l'API ne répond pas (code 200), le paquet de transaction continue d'augmenter et ne se vide que quand une sanction a été donnée.
   Les transactions qui ont été évaluées sont stockées dans une base NeonDB.
<img width="491" height="163" alt="image" src="https://github.com/user-attachments/assets/07af54af-0fe1-4242-8eb0-6297df99e87c" />

### III. Visualisation des résultats
   #### Streamlit

Les données sont exposées via un serveur Streamlit hébergé sur HuggingFace
https://synaxio-dashboard.hf.space
<img width="635" height="524" alt="image" src="https://github.com/user-attachments/assets/41925c6b-9cfa-497a-970d-1a2fe622c5f6" />


## Structure du dossier
<pre markdown="1">
│
├── .github/
│   └── workflows/
│       └── build_and_deploy.yml        # configuration GitAction  
│
├── Huggingface/                        # contenu des spaces sur Hugging face
│   ├── MLFlow-Server/                  # serveur MLflow
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── Streamlit/                      # app streamlit pour la visualisation des données
│       ├── Dockerfile
│       ├── app.py
│       └── requirements.txt
│
├── docker/
│   └── Dockerfile                      # Dockerfile spécifique utilisé par GitActions, couplé avec requirements.txt à la racine
│  
├── docker_automate/                    # Contient une app qui va simuler la consommation régulière de l'API
│   ├── __init__.py
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│  
├── etl/                                # Les fonctions qui permettent de se connecter à l'API qui fournit les transactions, au S3, à la base Neon. Simule un etl basique
│   ├── __init__.py
│   ├── app.py
│   ├── data_save.py
│   ├── Dockerfile
│   ├── exctract.py
│   ├── requirements.txt
│   └── transform.py
│
├── monitoring/                          # monitoring des modèles (TODO)
│   ├── __init__.py
│   ├── drift_monitor.py
│   └── scoring.py
│
├── tests/                               # Tests du code (A compléter)
│   ├── __init__.py
│   ├── test_api_etl.py
│   ├── test_etl.py
│   ├── test_model.py
│   └── test_training.py
│
├── training/                            # Entrainement du modèle
│   ├── __init__.py
│   ├── analyse.py                       # Choix de modèle, comparaison, grid search...
│   ├── data_loader.py                   # données d'entrainement
│   ├── preprocessing.py                 # pipelilne préparation
│   └── train.py                         # entrainement du modèle et enregistrement MLFlow
│
├── README.md
└── requirements.txt
</pre>  

## Installation

MLFlow & Streamlit: hébergé sur Huggingface dans des Spaces séparés dans lequels il suffit d'intégrer les credentials nécessaire pour se connecter sur le bucket S3 et la base NeonDB

api: un espace est créé sur huggingface, mais c'est gitactions qui va créer les éléments nécessaires au dépôt en fonction du modèle retenu

training: lancé directement depuis VSCode, nécessite les crédentials pour MLFlow

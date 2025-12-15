# JEDHA - Projet final - CI/CD fraud detection

## Enjeux du projet 
Mettre en oeuvre un pipeline de détection de fraude en temps réel
Ce projet comporte 3 parties:
  - Entrainement d'un modèle de prévision de fraude
  - Exploitation de ce modèle
  - Reporting sur les résultats

### I. Entrainement du modèle
  #### Modèle local
  - Récupération des données d'entrainement depuis un bucket S3
  - Sauvegarde des données brutes dans un bucket S3
  - Préparation des données
  - Entrainement du modèle (ici Random Forest)
  - Stockage du modèle et de ses métrics via un serveur MLFlow hébergé sur HuggingFace, qui sur un <br>
  enregistre les données dans un bucket S3 et une base Postresql (NeonDB)
<img width="2053" height="436" alt="image" src="https://github.com/user-attachments/assets/d56086d1-209b-4b35-85fc-03cf543aa8f0" />












### II. Prévision de fraudes
   #### ETL
   Les trois étapes de l'ETL sont encapsulées dans un script Python executé dans un Docker local 
   
<table>
  <tr>
    <td>Extract</td>
    <td>
      Les données d'apprentissage sont extraite d'un bucket S3 <br> 
      https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv <br>
      et directement transformées en un Pandas DataFrame
    </td>
  </tr>
  <tr>
    <td>Transform</td>
    <td>
      Les transactions récupérées sont formatées pour correspondre au format de l'API transaction <br> 
      https://sdacelo-real-time-fraud-detection.hf.space/current-transactions <br>
      Le dernier modèle fonctionnel sur MLFlow est récupéré et appliqué aux transactions
    </td>
  </tr>
  <tr>
    <td>Load</td>
    <td>
      Les données de transactions, enrichies des prédictions sont sauvegardées à la fois sur S3 et <br>
      sur la base PostGreSQL (NeonDB)
    </td>
  </tr>
</table>
<img width="911" height="462" alt="image" src="https://github.com/user-attachments/assets/aea37159-c774-49a3-8646-eba57e273587" />


### III. Visualisation des résultats
   #### Streamlit

Les données sont exposées via un serveur Streamlit hébergé sur HuggingFace
https://synaxio-dashboard.hf.space
<img width="635" height="524" alt="image" src="https://github.com/user-attachments/assets/41925c6b-9cfa-497a-970d-1a2fe622c5f6" />



## Architecture
<img width="362" height="919" alt="image" src="https://github.com/user-attachments/assets/bc324f83-0895-440f-acb3-6a923be8eaa1" />


## Structure du dossier
<pre markdown="1">
│
├── Huggingface/
│   ├── MLFlow-Server/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── Streamlit/
│       ├── Dockerfile
│       ├── app.py
│       └── requirements.txt
│  
├── etl/
│   ├── __init__.py
│   ├── app.py
│   ├── data_save.py
│   ├── Dockerfile
│   ├── exctract.py
│   ├── requirements.txt
│   └── transform.py
│
├── training/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── train.py
│
├── README.md
└── requirements.txt
</pre>  

## Installation

MLFlow & Streamlit: hébergé sur Huggingface dans des Spaces séparés dans lequels il suffit d'intégrer les credentials nécessaire pour se connecter sur le bucket S3 et la base NeonDB

etl: fonctionne dans un docker local, a juste besoin des credentials pour connection à S3, NeondDB et le serveur MLFlow

training: lancé directement depuis VSCode, nécessite les crédentials pour MLFlow

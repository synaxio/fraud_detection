
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")





def load_data(undersampling_ratio = 0):
    try:
        df = pd.read_csv("https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv", index_col=0)
    except Exception as e:
        print("   Could not load data from URL, loading local copy...")
        df = pd.read_csv("../data/fraudTest.csv", index_col=0)  

    df['current_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['trans_date_trans_time', 'unix_time'])

    # Convertion des int en float pour gérer les nan
    df = df.astype({col: "float64" for col in df.select_dtypes("int").columns})

    if undersampling_ratio >= 1:
    # Undersampling
        count_class_0, count_class_1 = df['is_fraud'].value_counts()

        df_class_0 = df[df['is_fraud'] == 0]
        df_class_1 = df[df['is_fraud'] == 1]

        # Sous-échantillonnage de la classe 0 (on la réduit au même nombre que la classe 1)
        df_class_0_under = df_class_0.sample(undersampling_ratio*count_class_1, random_state=42)

        # Concaténer les deux classes équilibrées
        df_balanced = pd.concat([df_class_0_under, df_class_1])

        # Mélanger les données pour avoir un ordre aléatoire
        df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df
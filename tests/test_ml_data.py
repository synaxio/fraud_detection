# import pandas as pd
# from evidently.test_suite import TestSuite
# from evidently.tests import *

# def test_data_quality_and_drift():
#     """
#     Tests Evidently pour vérifier la qualité des données et le drift.
#     """

#     reference = pd.read_csv("tests/reference_data.csv")  # ton dataset de référence
#     current = pd.read_csv("tests/current_data.csv")      # ton dataset le plus récent

#     suite = TestSuite(tests=[
#         # --- Qualité des données ---
#         TestNumberOfColumns(),
#         TestNumberOfRows(),
#         TestColumnsType(),
#         TestMissingValues(),

#         # --- Drift ---
#         TestShareOfDriftedColumns(),
#         TestColumnDrift(column_name="amt"),
#         TestColumnDrift(column_name="category"),
#         TestColumnDrift(column_name="city_pop"),

#         # --- Tests statistiques ---
#         TestColumnQuantile(column_name="amt", quantile=0.95),
#         TestValueRange(column_name="lat"),
#         TestValueRange(column_name="long"),
#     ])

#     suite.run(reference_data=reference, current_data=current)

#     # Si un test échoue → CI/CD stoppe
#     suite.raise_for_errors()


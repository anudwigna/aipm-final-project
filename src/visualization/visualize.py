import shap
import joblib
import os
import pandas as pd

project_path = os.path.dirname(os.path.abspath("requirements.txt"))
the_model = joblib.load(f"{project_path}/models/gb_model.joblib")
explainer = shap.Explainer(the_model)

X_train = pd.read_parquet(f"{project_path}/models/x_train.parquet")

instance_idx = 0
instance = X_train.iloc[instance_idx]  
shap_values = explainer(instance)

shap_values_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
shap.summary_plot(shap_values_df.values, shap_values_df.columns)


shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train)
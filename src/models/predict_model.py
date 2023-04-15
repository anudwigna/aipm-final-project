import joblib
import os
from src.models.train_model import test_df

project_path = os.path.dirname(os.path.abspath("requirements.txt"))
gb_model = joblib.load(f"{project_path}/models/gb_model.joblib")

test_preds = gb_model.predict(test_df)

print(test_preds)
import joblib

cat_models = joblib.load("cat_models.pkl")
xgb_models = joblib.load("xgb_models.pkl")

print(cat_models.keys())
print(xgb_models.keys())

print(cat_models.keys() == xgb_models.keys())
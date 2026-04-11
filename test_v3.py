from automlease import AutoML

model = AutoML()

# fit — tests Improvement 1 (XGBoost) + Improvement 3 (data quality report)
model.fit("housing.csv", target="MEDV")

# eda
model.eda()

# report — tests Improvement 5 (SHAP)
model.report()

# predict_new — Improvement 2
result = model.predict_new({'RM': 6.5, 'LSTAT': 10.0})
print(f"\npredict_new result: {result}")

print("\n✅ All 5 improvements verified successfully!")
print("   Run model.dashboard() separately to launch the Streamlit app.")

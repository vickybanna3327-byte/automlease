# automlease 🤖

**Automatic Machine Learning for absolute beginners.**
Train a machine learning model in just 3 lines of Python code!

## Installation
```bash
pip install automlease
```

## Quick Start
```python
from automlease import AutoML

model = AutoML()
model.fit('your_data.csv', target='target_column')
model.report()
```

## What it does automatically

- ✅ Loads any CSV file
- ✅ Cleans missing values
- ✅ Converts text columns to numbers
- ✅ Detects if task is classification or regression
- ✅ Trains multiple ML models and picks the best one
- ✅ Shows a detailed evaluation report
- ✅ Creates feature importance chart
- ✅ Saves the best model automatically

## Example — Heart Disease Prediction
```python
from automlease import AutoML

model = AutoML()
model.fit('heart_disease.csv', target='target')
model.report()
```

## Built With

- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- rich

## Author

**Vikash Singh Rajput**
BTech Management Graduate — NAIT, Edmonton, Canada
GitHub: https://github.com/vickybanna3327-byte

## License

MIT License
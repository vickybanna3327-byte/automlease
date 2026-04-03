from automlease import AutoML

model = AutoML()
model.fit('heart_disease_uci.csv', target='num')
model.report()
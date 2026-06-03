import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data= pd.read_csv('data/house_data.csv')

X= data[['area']]
y= data['price']

model= LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')

print("Model trained and saved as model.pkl")
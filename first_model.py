import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# dropna drops missing values (think af na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price
# print(melbourne_data.columns)
# print(y)
# print(y.shape)

melbourne_features = [
        'Rooms',
        'Bathroom',
        'Landsize',
        'Lattitude',
        'Longtitude',
        ]

X = melbourne_data[melbourne_features]
# print(X)
# print(X.shape)

melbourne_model = DecisionTreeRegressor(random_state=1)
print('training...')
melbourne_model.fit(X, y)
print('done')
print()


print('Making predictions for the following 5 houses')
print(X.head())
print()

print('The predictions are')
print(melbourne_model.predict(X.head()))
print()

print('Real prices')
print(y.head())
print()

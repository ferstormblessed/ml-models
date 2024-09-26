import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

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

trainX, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

melbourne_model = DecisionTreeRegressor(random_state=1)
print('training...')
melbourne_model.fit(trainX, train_y)
print('done')
print()

# print('Making predictions for the following 5 houses')
# print(X.head())
# print()

print('The predictions are')
val_predictions = melbourne_model.predict(val_X)
print(val_predictions)
print()

# print('Real prices')
# print(y.head())
# print()

print(mean_absolute_error(val_y, val_predictions))

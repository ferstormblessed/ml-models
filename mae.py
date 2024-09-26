import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# load data
melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing price values
filtererd_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtererd_melbourne_data.Price
melbourne_features = [
        'Rooms',
        'Bathroom',
        'Landsize',
        'Lattitude',
        'Longtitude',
        ]
X = filtererd_melbourne_data[melbourne_features]

# Define model
melbouren_model = DecisionTreeRegressor()
# Fit model
melbouren_model.fit(X, y)

predicted_home_prices = melbouren_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

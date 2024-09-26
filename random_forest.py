from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Data loading
melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# dropna drops missing values (think af na as "not available")
filtered_melbourne_data = melbourne_data.dropna(axis=0)
y = filtered_melbourne_data.Price

melbourne_features = [
        'Rooms',
        'Bathroom',
        'Landsize',
        'Lattitude',
        'Longtitude',
        ]

X = filtered_melbourne_data[melbourne_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

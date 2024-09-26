from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


def get_mae(max_lead_nodes, train_X, val_X, train_y, val_y):
    forest_model = RandomForestRegressor()
    forest_model.fit(train_X, train_y)
    melb_preds = forest_model.predict(val_X)
    mae = mean_absolute_error(val_y, melb_preds)
    return (mae)


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

for max_leaf_nodes in [450, 460, 470, 480, 490, 500, 510, 520]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" % (max_leaf_nodes, my_mae))

for max_leaf_nodes in [450, 452, 454, 456, 458, 460, 462, 464]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" % (max_leaf_nodes, my_mae))

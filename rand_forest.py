# out-of-sample data and corresponding mean absolute error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# drop (remove) rows with missing data
melbourne_data = melbourne_data.dropna(axis=0) 

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude',
 'Longtitude']
X = melbourne_data[melbourne_features]
y = melbourne_data.Price

# split data into training and validation data, for both features and target
# The split is based on a random number generator. 
# Supplying a numeric value to the random_state argument guarantees
#       we get the same split every time we run this script

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define model
melbourne_model = RandomForestRegressor(random_state=1)

# fit model
melbourne_model.fit(train_X, train_y)

# predict some prices
predicted_model = melbourne_model.predict(val_X)

# Create function to define several models
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes,
                 random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

print("\nmae with 5 leaf nodes:", 
            get_mae(5, train_X, val_X, train_y, val_y))
#print("\nmae with 50 leaf nodes:", 
    #        get_mae(50, train_X, val_X, train_y, val_y))
#print("mae with 500 leaf nodes:", 
    #        get_mae(500, train_X, val_X, train_y, val_y))
#print("mae with 5000 leaf nodes:", 
            #get_mae(5000, train_X, val_X, train_y, val_y))
print("mae with 50000 leaf nodes:", 
            get_mae(50000, train_X, val_X, train_y, val_y))
#####
print("mae no node specified:", mean_absolute_error(val_y, predicted_model))

####Output
#mae no node specified: 207190.6873773146
#mae with 5 leaf nodes: 375038.0474733729
#mae with 500 leaf nodes: 208580.49362309175
#mae with 5000 leaf nodes: 207143.62285376125
#mae with 50000 leaf nodes: 207143.62285376125

# Random forest regression without a node selected
#    will give a prediction (mae 207190, no nodes specified)
#       as close to the best prediction (mae 207143)
#            with 500 leaf nodes selected




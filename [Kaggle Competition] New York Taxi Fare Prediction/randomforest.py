import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.3f' % x)
RSEED = 100


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)

data = pd.read_csv('train_1.csv', nrows = 100000, parse_dates = ['pickup_datetime']).drop(columns = 'key')
data = data.dropna()
data.drop(['pickup_datetime'],1,inplace=True)


data = data[data['fare_amount'].between(left = 2.5, right = 100)]

data['fare-bin'] = pd.cut(data['fare_amount'], bins = list(range(0, 50, 5))).astype(str)
data.loc[data['fare-bin'] == 'nan', 'fare-bin'] = '[45+]'
data.loc[data['fare-bin'] == '(5, 10]', 'fare-bin'] = '(05, 10]'


data = data.loc[data['passenger_count'] <= 8]


data = data.loc[data['pickup_latitude'].between(40, 42)]
data = data.loc[data['pickup_longitude'].between(-75, -72)]
data = data.loc[data['dropoff_latitude'].between(40, 42)]
data = data.loc[data['dropoff_longitude'].between(-75, -72)]

def radian_conv(degree):
    return  np.radians(degree)  

#HAVERSINE DISTANCE
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])

    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))

def add_airport_dist(dataset):
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892,-74.0445) # Statue of Liberty
    nyc_coord = (40.7141667,-74.0063889) 
    
    
    pickup_lat = dataset['pickup_latitude']
    dropoff_lat = dataset['dropoff_latitude']
    pickup_lon = dataset['pickup_longitude']
    dropoff_lon = dataset['dropoff_longitude']
    
    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)
    pickup_sol = sphere_dist(pickup_lat, pickup_lon, sol_coord[0], sol_coord[1]) 
    dropoff_sol = sphere_dist(sol_coord[0], sol_coord[1], dropoff_lat, dropoff_lon)
    pickup_nyc = sphere_dist(pickup_lat, pickup_lon, nyc_coord[0], nyc_coord[1]) 
    dropoff_nyc = sphere_dist(nyc_coord[0], nyc_coord[1], dropoff_lat, dropoff_lon)
    
    
    
    dataset['jfk_dist'] = pickup_jfk + dropoff_jfk
    dataset['ewr_dist'] = pickup_ewr + dropoff_ewr
    dataset['lga_dist'] = pickup_lga + dropoff_lga
    dataset['sol_dist'] = pickup_sol + dropoff_sol
    dataset['nyc_dist'] = pickup_nyc + dropoff_nyc
    
    return dataset

data = add_airport_dist(data)

data['abs_lat_diff'] = (data['dropoff_latitude'] - data['pickup_latitude']).abs()
data['abs_lon_diff'] = (data['dropoff_longitude'] - data['pickup_longitude']).abs()


def minkowski_distance(x1, x2, y1, y2, p):
 return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)

data['euclidean'] = minkowski_distance(data['pickup_longitude'], data['dropoff_longitude'], data['pickup_latitude'], data['dropoff_latitude'], 2)
data['haversine'] = sphere_dist(data['pickup_latitude'], data['pickup_longitude'], data['dropoff_latitude'] , data['dropoff_longitude']) 

test = pd.read_csv('test.csv', parse_dates = ['pickup_datetime'])
test.drop(['pickup_datetime'],1,inplace=True)

test['abs_lat_diff'] = (test['dropoff_latitude'] - test['pickup_latitude']).abs()
test['abs_lon_diff'] = (test['dropoff_longitude'] - test['pickup_longitude']).abs()
test_id = list(test.pop('key'))
test['euclidean'] = minkowski_distance(test['pickup_longitude'], test['dropoff_longitude'],test['pickup_latitude'], test['dropoff_latitude'], 2)
test = add_airport_dist(test)
test['haversine'] = sphere_dist(test['pickup_latitude'], test['pickup_longitude'], test['dropoff_latitude'] , test['dropoff_longitude'])

def metrics(train_pred, valid_pred, y_train, y_valid):
 train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
 valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
 train_ape = abs((y_train - train_pred) / y_train)
 valid_ape = abs((y_valid - valid_pred) / y_valid)
 train_ape[train_ape == np.inf] = 0
 train_ape[train_ape == -np.inf] = 0
 valid_ape[valid_ape == np.inf] = 0
 valid_ape[valid_ape == -np.inf] = 0
 train_mape = 100 * np.mean(train_ape)
 valid_mape = 100 * np.mean(valid_ape)
 return train_rmse, valid_rmse, train_mape, valid_mape

def evaluate(model, features, X_train, X_valid, y_train, y_valid):
 train_pred = model.predict(X_train[features])
 valid_pred = model.predict(X_valid[features])
 train_rmse, valid_rmse, train_mape, valid_mape = metrics(train_pred, valid_pred,y_train, y_valid)
 print('Training:rmse = ' + str(round(train_rmse, 2))+ '\t mape = ' + str(round(train_mape,2)))
 print('Validation: rmse = ' + str(round(valid_rmse, 2))+ '\t mape = ' + str(round(valid_mape,2)))


X_train, X_valid, y_train, y_valid = train_test_split(data, np.array(data['fare_amount']),stratify = data['fare-bin'],random_state = RSEED, test_size= 8000)

random_forest = RandomForestRegressor(n_estimators = 20, max_depth = 20, max_features = None, oob_score = True, bootstrap = True, verbose = 1, n_jobs = -1)
random_forest.fit(X_train[['euclidean', 'abs_lat_diff', 'abs_lon_diff', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist', 'nyc_dist', 'haversine']], y_train)
evaluate(random_forest, ['euclidean', 'abs_lat_diff', 'abs_lon_diff', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist', 'nyc_dist', 'haversine'], X_train, X_valid, y_train, y_valid)

preds = random_forest.predict(test[['euclidean', 'abs_lat_diff', 'abs_lon_diff', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist', 'nyc_dist', 'haversine']])
sub = pd.DataFrame({'key': test_id, 'fare_amount': preds})
sub.to_csv('sub_rf_simple.csv', index = False)












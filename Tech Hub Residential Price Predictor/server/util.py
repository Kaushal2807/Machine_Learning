import json
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

__location = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __location

def load_data():
    print('loading saved data....start')
    global __data_columns
    global __location
    global __model

    with open("k:/Machine_Learning/Tech Hub Residential Price Predictor/model/columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __location = __data_columns[3:]

    with open("k:/Machine_Learning/Tech Hub Residential Price Predictor/model/banglore_home_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)

if __name__ == '__main__':
    load_data()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location




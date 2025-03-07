import pandas as pd
import numpy as np
import warnings



def load_data():
    test = pd.read_csv('proj1/source/test.csv')
    train = pd.read_csv('proj1/source/train.csv')
    data = pd.concat([test, train])
    data = data.sample(129880).reset_index().drop(['index', 'id'], axis=1)
    data = data.drop(['Unnamed: 0', 'Arrival Delay in Minutes', 'Departure Delay in Minutes'], axis=1)
    return data



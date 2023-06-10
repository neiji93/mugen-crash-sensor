#!/usr/bin/env python3
import pandas as pd
from sklearn.neural_network import MLPClassifier
import random
import pickle


col_features = [ 'total_damage_normed'\
               , 'total_spark_normed'\
               , 'priority_val_normed']

df = pd.read_csv("processed_data.csv")
x_raw = [df[col].to_list() for col in col_features]
X = list(map(list, zip(*x_raw)))
y = df['frozen'].to_list()

with open('freeze_network.sav', 'rb') as m:
    net = pickle.load(m)

total_accuracy = net.score(X, y)
print(f"The total accuracy is {total_accuracy:0.3e}")

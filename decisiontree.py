import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import pydotplus

dataset = pd.read_csv('wine.data', header=None)
dataset.columns = ['label',
                   'alcohol', 
                   'malic_acid', 
                   'ash', 
                   'alcalinity_of_ash', 
                   'magnesium', 
                   'total_phenols', 
                   'flavanoids', 
                   'nonflavanoid_phenols', 
                   'proanthocyanins', 
                   'color_intensity', 
                   'hue',
                   'OD280/OD315',
                   'proline'
                   ]
print(dataset.head())

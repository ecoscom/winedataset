import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from IPython.display import Image

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

x = dataset.values[:, 1:]
y = dataset.values[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

scaler = StandardScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test = scaler.transform(x_test)


def train_model(height):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=height, random_state=0)
    model.fit(x_train, y_train)
    return model

'''for height in range(1,21):
    model = train_model(height)
    y_pred = model.predict(x_test)
    
    print('-'*10+'/n')
    print(f'Altura - {height}\n')
    print('Precisão:' + str(accuracy_score(y_test, y_pred)))
'''



model = train_model(3)


feature_names = ['alcohol',
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
                 'proline']

classes_names = ['%.f' % i for i in model.classes_]

dot_data = export_graphviz(model, filled=True, feature_names=feature_names, class_names=classes_names, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())
graph.write_png("tree.png")
Image('tree.png')



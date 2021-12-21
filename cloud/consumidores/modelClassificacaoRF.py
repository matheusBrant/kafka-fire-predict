from matplotlib import pyplot as plt, rc_file
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from csv import reader
from sklearn import metrics
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

dataset_forestfire = pd.read_csv('../dados/dadosClassificacao/forest_fire_classificacao0-1.csv')  
dataset_forestfire= dataset_forestfire.drop(['area'],axis=1) #para dropar a area
target = dataset_forestfire.pop('fire')

X_train, X_test, y_train, y_test = train_test_split(dataset_forestfire, target, test_size=0.2) # 70% training and 30% test

rf=RandomForestClassifier(n_estimators=25)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
print("Acurácia RF:",round(metrics.accuracy_score(y_test, y_pred)*100, 2),"%")
print("Precisão RF: ", round(recall_score(y_test, y_pred)*100, 2),"%")

#rf.fit(dataset_forestfire, target)
print(rf.predict([[6,5,92.5,121.1,674.4,8.6,25.1,27,4.0,0.0]]))
                
def predictRF(row):
    predict = rf.predict([row])
    if predict == 0:
        print(row, ' ---> Chance de fogo baixa 🌳✅')
    elif predict == 1: 
        print(row, ' ---> Chance de fogo alta 🌳🔥')
    '''elif predict == 2: 
        print(row, ' ---> Chance de fogo alta: até 1 hectares 🌳🔥🔥')
    elif predict == 3: 
        print(row, ' ---> Chance de fogo alta: até 2 hectares 🌳🔥🔥🔥')
    elif predict == 4: 
        print(row, ' ---> Chance de fogo alta: até 20 hectares 🌳🔥🔥🔥🔥')'''
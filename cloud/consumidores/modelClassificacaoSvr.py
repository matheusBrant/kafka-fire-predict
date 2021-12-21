from matplotlib import pyplot as plt, rc_file
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from csv import reader
from sklearn import metrics
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error # mesma funcao do MSE


warnings.filterwarnings("ignore")

dataset_forestfire = pd.read_csv('../dados/dadosClassificacao/forest_fire_classificacao.csv')  
dataset_forestfire= dataset_forestfire.drop(['fire'],axis=1) #para dropar a area
#dataset_forestfire= dataset_forestfire.drop(['month'],axis=1) #para dropar a area
#dataset_forestfire= dataset_forestfire.drop(['day'],axis=1) #para dropar a area

target = dataset_forestfire.pop('area')

X_train, X_test, y_train, y_test = train_test_split(dataset_forestfire, target, test_size=0.35,random_state=42) # 70% training and 30% test


regressor = SVR(kernel='rbf', gamma = 10)
regressor.fit(X_train, y_train)
y_pred  = regressor.predict(X_test)

RMSE = mean_squared_error(y_test,y_pred,squared=False) 
# argumento 'squared' dado como false nos da o RMSE

# ou podemos simplesmente tirar a raiz quadrada do MSE
RMSE = RMSE**0.5

print(RMSE)

def predictSVR(row):
    predict = regressor.predict([row])
    print(row, '--->', predict)

'''def predictDT(row):
    predict = dt.predict([row])
    if predict == 0:
        print(row, ' ---> Chance de fogo baixa 🌳✅')
    elif predict == 1: 
        print(row, ' ---> Chance de fogo alta: 0.1+ hectares 🌳🔥')
    elif predict == 2: 
        print(row, ' ---> Chance de fogo alta: 10+ hectares 🌳🔥🔥')
    elif predict == 3: 
        print(row, ' ---> Chance de fogo alta: 50+ hectares 🌳🔥🔥🔥')
    elif predict == 4: 
        print(row, ' ---> Chance de fogo alta: 200+ hectares 🌳🔥🔥🔥🔥')'''
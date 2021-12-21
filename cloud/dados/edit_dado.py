import pandas as pd

csv_input = pd.read_csv('dadosRegressao/dado_recebido.csv')
for i in range(len(csv_input)): 
    if csv_input['area'][i] == 0.0:
        csv_input['month'][i] = int(0)
    elif csv_input['area'][i] <= 0.5:
        csv_input['month'][i] = int(1)
    elif csv_input['area'][i] <= 1.0:
        csv_input['month'][i] = int(2)
    elif csv_input['area'][i] <= 2.0:
        csv_input['month'][i] = int(3)
    elif csv_input['area'][i] <= 20.0:
        csv_input['month'][i] = int(4)
    elif csv_input['area'][i] <= 50.0:
        csv_input['month'][i] = int(5)
    elif csv_input['area'][i] <= 100.0:
        csv_input['month'][i] = int(6)
    elif csv_input['area'][i] <= 200.0:
        csv_input['month'][i] = int(7)
    elif csv_input['area'][i] <= 400.0:
        csv_input['month'][i] = int(8)
    elif csv_input['area'][i] <= 600.0:
        csv_input['month'][i] = int(9)
    elif csv_input['area'][i] <= 800.0:
        csv_input['month'][i] = int(10)
    else:
        csv_input['month'][i] = int(11)

csv_input = csv_input.rename(columns={'month': 'fire'})
csv_input= csv_input.drop(['day'],axis=1)
#csv_input= csv_input.drop(['fire'],axis=1) #ativar para tirar o fire para test
#csv_input = csv_input[['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area','fire']] 
csv_input = csv_input[['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area','fire']] #para tirar a area e o fire
csv_input.to_csv('dadosClassificacao/out2.csv',  index=False)


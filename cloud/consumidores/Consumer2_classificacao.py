from confluent_kafka import Consumer, KafkaException
from modelClassificacaoSvm import predictSVM
from modelClassificacaoRF import predictRF
from modelClassificacaoDT import predictDT
import certifi
import sys, csv, os


if __name__ == '__main__':

  topic = "fire-predict"  
  conf = {  
    'group.id': "fire-predict", 
    'session.timeout.ms': 6000,
    'auto.offset.reset': 'latest',
    'bootstrap.servers': 'cell-1.streaming.sa-saopaulo-1.oci.oraclecloud.com:9092',  
    'security.protocol': 'SASL_SSL',  
    'ssl.ca.location': certifi.where(),    
    'sasl.mechanism': 'PLAIN',  
    'sasl.username': 'matheusbrant/oracleidentitycloudservice/matheusbrantgo@gmail.com/ocid1.streampool.oc1.sa-saopaulo-1.amaaaaaaz2nkdgaatblh5dkumqibancjusgaghu24vhrec4yvhacwhrbixta',  # from step 2 of Prerequisites section
    'sasl.password': '2P0vj>Fxh4ghGe.KHD:p',
   }  

consumer = Consumer(conf)
consumer.subscribe([topic])

modelo = input("\nQual será o método de predição: \n a -> SVM\n b -> RandomForest\n c -> DecisionTree\n--> ")


fileName = r"'../dados/dadosClassificacao/forest_fire_classificacao_predict.csv'"
if os.path.isfile(fileName) == True:
    os.remove('../dados/dadosClassificacao/forest_fire_classificacao_predict.csv')
else:
    print('--> Aguardando dados... ⏳')

try:
    with open('../dados/dadosClassificacao/forest_fire_classificacao_predict.csv', 'w', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow(["X","Y","FFMC","DMC","DC","ISI","temp","RH","wind","rain","area"])

    while True:
        print('--> Aguardando dados... ⏳')

        msg = consumer.poll(timeout=10.0)
        if msg is None:
            continue
        if msg.error():
            raise KafkaException(msg.error())
        else:
            # Proper message
            sys.stderr.write('%% %s [%d] do offset %d, chaveado a key: %s:\n' %
                                (msg.topic(), msg.partition(), msg.offset(),
                                str(msg.key().decode('utf-8'))))

            with open('../dados/dadosClassificacao/forest_fire_classificacao_predict.csv', 'w', newline='\n') as file:

                writer = csv.writer(file)
                output=msg.value().decode('utf-8')
                strin=eval(output)
                writer.writerows([strin])

                #aplicando o modelo
                #parametro: linha da entrada para predicao do classificador

                if modelo == 'a':
                    predictSVM(strin)
                elif modelo == 'b':
                    predictRF(strin)
                elif modelo == 'c':
                    predictDT(strin)
                

                bt=1000000
                mega=2
                #limite maximo para nao travar meu pc
                if mega > 5:
                    print('-> Limite de arquivo excedido <-')
                    os.remove('../dados/dadosClassificacao/forest_fire_classificacao_predict.csv')
                    break
                tam_mb = bt*mega
                if os.stat('../dados/dadosClassificacao/forest_fire_classificacao_predict.csv').st_size >= tam_mb:
                    print('------------------------------------------- \n -> Arquivo chegou no limite especificado <-')
                    break

except KeyboardInterrupt:
    sys.stderr.write('%% \n--> Interrompido pelo usuário ❌\n')

finally:
    consumer.close()
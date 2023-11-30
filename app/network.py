import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import random_split, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

datos = pd.read_csv("static/files/Churn_Modelling.csv")
datos.head()
#Separamos la última columna para que sea variable destino
datos_y = datos[datos.columns[-1]]
datos_y.head()

#Se eliminan las columnas que no funcionarán
datos_x = datos.drop(columns=["RowNumber","CustomerId","Surname","Exited"])
datos_x.head()


#las redes neuronales no procesan bien el texto, por lo tanto los datos
#de algunas columnas deben pasar a ser números
#Covertimos en one hot encoding las columnas de genero y zona geográfica
datos_x=pd.get_dummies(datos_x)
datos_x.head()
#print(str(datos_x.head()))

#ESCALANDO DATOS
#Escalamos los valores para que esten dentro de un rango mas corto
escalador=StandardScaler()
datos_x=escalador.fit_transform(datos_x)


#DIVIDIR DATOS ENTRE ENTRENAMIENTO Y TEST
#print(str(datos_x.shape[0]))

x_train, x_test, y_train, y_test = train_test_split(datos_x, datos_y, test_size=0.2, random_state=2)
#print("X train: {}, X Test:{}, y_train:{}, y_test:{}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

n_entradas=x_train.shape[1]

#TENSORES
#Para poder procesar los datos en la red reuronal necesitamos que todos los datos entén en tensores, así que haremos las conversaciones necesarias
t_x_train = torch.from_numpy(x_train).float().to("cpu") #MPS
t_x_test = torch.from_numpy(x_test).float().to("cpu")
t_y_train = torch.from_numpy(y_train.values).float().to("cpu")
t_y_test = torch.from_numpy(y_test.values).float().to("cpu")
t_y_train = t_y_train[:,None]
t_y_test = t_y_test[:,None]

test = TensorDataset(t_x_test,t_y_test)
#print(test[0])

class Red(nn.Module):
    def __init__(self, n_entradas):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_entradas, 15)
        self.linear2 = nn.Linear(15, 8)
        #self.linear3 = nn.Linear(8,160)
        #self.linear4 = nn.Linear(160,200)
        #self.linear5 = nn.Linear(200,1)
        self.linear3 = nn.Linear(8, 1)

    def forward(self, inputs):
        pred_1 = torch.sigmoid(input=self.linear1(inputs))
        pred_2 = torch.sigmoid(input=self.linear2(pred_1))
        #prediction = torch.sigmoid(input=self.linear3(prediction))
        #prediction = torch.sigmoid(input=self.linear4(prediction))
        #prediction = torch.sigmoid(input=self.linear5(prediction))
        pred_f = torch.sigmoid(input=self.linear3(pred_2))
    
        return pred_f


lr = 0.001
epochs = 2000
estatus_print = 100

model = Red(n_entradas=n_entradas)
#print(model.parameters())
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
print("Arquitectura del modelo: {}".format(model))
historico = pd.DataFrame()

print("Entrenando el modelo")
for epoch in range(1, epochs+1):
    y_pred  = model(t_x_train)
    #print(y_pred.shape)
    loss = loss_fn(input=y_pred, target=t_y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % estatus_print == 0:
        print(f"\nEpoch {epoch} \t Loss: {round(loss.item(), 4)}")
    
    with torch.no_grad():
        y_pred = model(t_x_test)
        y_pred_class = y_pred.round()
        correct = (y_pred_class == t_y_test).sum()
        accuracy = 100 * correct / float(len(t_y_test))
        if epoch % estatus_print == 0:
            print("Acuracy: {}".format(accuracy.item()))
    
    df_tmp = pd.DataFrame(data={
        'Epoch': epoch,
        'Loss': round(loss.item(), 4),
        'Accuracy': round(accuracy.item(), 4)
    }, index=[0])
    historico = pd.concat(objs=[historico, df_tmp], ignore_index=True, sort=False)

#print("Accuracy final: {}".format(round(accuracy.item(), 4)))
"""
plt.figure(figsize=(10, 10))
plt.plot(historico['Epoch'], historico['Loss'], label='Loss')
plt.title("Loss", fontsize=25)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid()
plt.show()
plt.savefig('line_plot.svg')  

"""
prediction = model(t_x_test[4])
print(prediction)
print(str(t_y_test[4]))
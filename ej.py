#---------------------------------------------  
# CARGA DE LAS OBSERVACIONES  
#---------------------------------------------  

import pandas as pnd
observaciones = pnd.read_csv("datas/sonar.all-data.csv") 

#Para el aprendizaje solo se toman los datos procedentes del sonar 
X = observaciones[observaciones.columns[0:60]].values

#Solo se toman los etiquetados  
y = observaciones[observaciones.columns[60]]

#Se codifica: Las minas son iguales a 0 y las rocas son iguales a 1  
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

#Se añade un cifrado para crear clases:  
# Si es una mina [1,0]  
# Si es una roca [0,1]  
import numpy as np
n_labels = len(y)
n_unique_labels = len(np.unique(y))
one_hot_encode = np.zeros((n_labels,n_unique_labels))
one_hot_encode[np.arange(n_labels),y] = 1
Y=one_hot_encode

#Verificación tomando los registros 0 y 97  
print("Clase Roca:",int(Y[0][1]))
print("Clase Mina:",int(Y[97][1])) 

#Mezclamos las observaciones  
from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=1)

#Creación de los conjuntos de aprendizaje y de las pruebas 
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.07, random_state=42) 

import tensorflow as tf

epochs = 300
cantidad_neuronas_entrada = 60
cantidad_neuronas_salida = 2
cantidad_neuronas_capa_oculta = 24
tasa_aprendizaje = 0.01

#Variable TensorFLow correspondiente a los 60 valores de las 
#neuronas de entrada  
tf_neuronas_entradas_X = tf.placeholder(tf.float32,[None, 60])

#Variable TensorFlow correspondiente a las 2 neuronas de salida  
tf_valores_reales_Y = tf.placeholder(tf.float32,[None, 2])


pesos = {'capa_entrada_hacia_oculta': tf.Variable(tf.random_uniform([60, 24], minval=-0.3, maxval=0.3),tf.float32), 'capa_oculta_hacia_salida':tf.Variable(tf.random_uniform([24, 2], minval=-0.3, maxval=0.3),tf.float32),}
peso_sesgo = {'peso_sesgo_capa_entrada_hacia_oculta':tf.Variable(tf.zeros([24]), tf.float32),'peso_sesgo_capa_oculta_hacia_salida': tf.Variable(tf.zeros([2]), tf.float32),} 


def red_neuronas_multicapa(observaciones_en_entradas, pesos,
peso_sesgo):

    #Cálculo de la activación de la primera capa  
    primera_activacion = tf.sigmoid(tf.matmul(tf_neuronas_entradas_X,pesos['capa_entrada_hacia_oculta']) +peso_sesgo['peso_sesgo_capa_entrada_hacia_oculta'])

    #Cálculo de la activación de la segunda capa  
    activacion_capa_oculta =tf.sigmoid(tf.matmul(primera_activacion,pesos['capa_oculta_hacia_salida']) +
    peso_sesgo['peso_sesgo_capa_oculta_hacia_salida'])

    return activacion_capa_oculta

red = red_neuronas_multicapa(tf_neuronas_entradas_X, pesos, peso_sesgo) 

#Función de error de media cuadrática MSE  
funcion_error = tf.reduce_sum(tf.pow(tf_valores_reales_Y-red,2))

#Descenso de gradiente con una tasa de aprendizaje fijada a 0,01  
optimizador =tf.train.GradientDescentOptimizer(learning_rate=tasa_aprendizaje).minimize(funcion_error) 

#Inicialización de las variables  
init = tf.global_variables_initializer()

#Inicio de una sesión de aprendizaje  
session = tf.Session()
session.run(init)

#Para la realización del gráfico para la MSE  
Grafica_MSE=[]


#Para cada epoch  
for i in range(epochs):

    #Realización del aprendizaje con actualización de los pesos  
    session.run(optimizador, feed_dict = {tf_neuronas_entradas_X:
    train_x, tf_valores_reales_Y:train_y})

    #Calcular el error  
    MSE = session.run(funcion_error, feed_dict =
    {tf_neuronas_entradas_X: train_x, tf_valores_reales_Y:train_y})

    #Visualización de la información  
    Grafica_MSE.append(MSE)
    print("EPOCH (" + str(i) + "/" + str(epochs) + ") - MSE: "+str(MSE))


#Visualización gráfico de la MSE  
import matplotlib.pyplot as plt
plt.plot(Grafica_MSE)
plt.ylabel('MSE')
plt.show() 


#Recuperación de los índices de las clasificaciones realizadas  

clasificaciones = tf.argmax(red, 1)

#Comparamos los índices procedentes de las clasificaciones con los 
#esperados para conocer la cantidad de clasificaciones correctas  

formula_calculo_clasificaciones_correctas = tf.equal(clasificaciones,tf.argmax(tf_valores_reales_Y,1))


#A continuación calculamos la precisión haciendo la media (tf.mean)  
#de las clasificaciones correctas (después de haberlas convertido en  

formula_precision =tf.reduce_mean(tf.cast(formula_calculo_clasificaciones_correctas,tf.float32)) 
#----------------------------------------------------------------  
# PRECISIÓN EN LOS DATOS DE PRUEBAS  
#----------------------------------------------------------------  

num_clasificaciones = 0
num_clasificaciones_correctas = 0

#Miramos todo el conjunto de los datos de pruebas (text_x)  
for i in range(0,test_x.shape[0]):

    datosSonar = test_x[i].reshape(1,60)
    clasificacionEsperada = test_y[i].reshape(1,2)

    # Hacemos la clasificación  
    prediction_run = session.run(clasificaciones,feed_dict={tf_neuronas_entradas_X:datosSonar})

    accuracy_run = session.run(formula_precision,feed_dict={tf_neuronas_entradas_X:datosSonar, tf_valores_reales_Y:clasificacionEsperada})

    print(i,"Clase esperada: ", int(session.run(tf_valores_reales_Y [i][1],feed_dict={tf_valores_reales_Y:test_y})), "Clasificación: ", prediction_run[0] )

    num_clasificaciones = num_clasificaciones+1
    if(accuracy_run*100 ==100):
        num_clasificaciones_correctas = num_clasificaciones_correctas+1
print("-------------")
print("Precisión en los datos de pruebas = "+str((num_clasificaciones_correctas/num_clasificaciones)*100)+"%") 

##24NEURONAS

pesos = {'capa_entrada_hacia_oculta':tf.Variable(tf.random_uniform([60, 12], minval=-0.3,maxval=0.3),tf.float32),
'capa_oculta_hacia_salida':tf.Variable(tf.random_uniform([12, 2], minval=-0.3, maxval=0.3),tf.float32),}

peso_sesgo = {'peso_sesgo_capa_entrada_hacia_oculta': tf.Variable(tf.zeros([12]), tf.float32),'peso_sesgo_capa_oculta_hacia_salida':tf.Variable(tf.zeros([2]), tf.float32),} 


#PARA MEJORES RESULTADOS

epochs = 600

pesos ={'capa_entrada_hacia_oculta':
tf.Variable(tf.random_normal([60, 26]), tf.float32),
'capa_oculta_hacia_salida' :
tf.Variable(tf.random_normal([26, 2]), tf.float32),
}

peso_sesgo = {

'peso_sesgo_capa_entrada_hacia_oculta':
tf.Variable(tf.zeros([26]), tf.float32),

'peso_sesgo_capa_oculta_hacia_salida':
tf.Variable(tf.zeros([2]), tf.float32),
} 

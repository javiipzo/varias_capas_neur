import tensorflow as tf
import numpy as np


#-------------------------------------  
#    DATOS DE APRENDIZAJE  
#-------------------------------------  

#Los datos se transforman en decimales  

valores_entradas_X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
valores_a_predecir_Y = [[0.], [1.], [1.], [0.]]



#-------------------------------------
#    PARÁMETROS DE LA RED  
#-------------------------------------  
#Variable TensorFLow correspondiente a los valores neuronas 
#de entrada 
tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])

#Variable TensorFlow correspondiente a la neurona de salida  

tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])


#Cantidad de neuronas en la capa oculta
nbr_neuronas_capa_oculta = 2

#PESOS  
#Los primeros están 4 : 2 en la entrada (X1 y X2) y 2  

pesos = tf.Variable(tf.random_normal([2, 2]), tf.float32)

#los pesos de la capa oculta están 2 : 2 en la entrada  

pesos_capa_oculta = tf.Variable(tf.random_normal([2, 1]),
tf.float32)

#El primer sesgo contiene 2 pesos  
sesgo = tf.Variable(tf.zeros([2]))

#El segundo sesgo contiene 1 peso
sesgo_capa_oculta = tf.Variable(tf.zeros([1]))

#después aplicación de la función sigmoide (tf.sigmoid)  
activacion = tf.sigmoid(tf.matmul(tf_neuronas_entradas_X, pesos) +sesgo)

activacion_capa_oculta = tf.sigmoid(tf.matmul(activacion, pesos_capa_oculta) + sesgo_capa_oculta)



#Función del error de media cuadrática MSE  
funcion_error = tf.reduce_sum(tf.pow(tf_valores_reales_Y-
activacion_capa_oculta,2))

#Descenso del gradiente con una tasa de aprendizaje fijada en 0,1  
optimizador = tf.train.GradienteDescensoOptimizer(learning_rate=0.1).minimize(funcion_error)

#Cantidad de epochs  
epochs = 100000

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
    session.run(optimizador, feed_dict = {tf_neuronas_entradas_X:valores_entradas_X, tf_valores_reales_Y:valores_a_predecir_Y})

    #Calcular el error  
    MSE = session.run(funcion_error, feed_dict =
    {tf_neuronas_entradas_X: valores_entradas_X,
    tf_valores_reales_Y:valores_a_predecir_Y})

    #Visualización de la información  
    Grafica_MSE.append(MSE)
    print("EPOCH (" + str(i) + "/" + str(epochs) + ") -MSE: "+
    str(MSE))


#Visualización gráfica  
import matplotlib.pyplot as plt
plt.plot(Grafica_MSE)
plt.ylabel('MSE')
plt.show()


session.close() 

print("--- VERIFICACIONES ----")

for i in range(0,4):
    print("Observación:"+str(valores_entradas_X[i])+ " - Esperado: "+str(valores_a_predecir_Y[i])+" - Predicción: "+str(session.run(activacion_capa_oculta,feed_dict={tf_neuronas_entradas_X: [valores_entradas_X[i]]}))) 

session.run(activacion_capa_oculta,feed_dict={tf_neuronas_entradas_X: [valores_entradas_X[i]]}) 
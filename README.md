# 1erParcialIA_Fiuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ruta= '/content/drive/MyDrive/Colab Notebooks/Mediciones.csv'
# Función para calcular los coeficientes de regresión manualmente
def regresion_manual(X, y):
    X = np.column_stack((np.ones(len(X)), X))  # Agregar una columna de unos para el término independiente
    coeficientes = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y  # Calcular los coeficientes utilizando la fórmula de la pseudo inversa
    return coeficientes #@ es para multiplicar matrices también. T es la transpuesta de X
# np.linalg.pinv(A).dot(Y)

# Función para predecir los valores de y
def predecir(X, coeficientes):
    Xm = np.column_stack((np.ones(len(X)), X))  # Agregar una columna de unos para el término independiente
    return Xm @ coeficientes
# Calcular métricas de evaluación manualmente
def rmse(y_true, y_pred):
    error = y_true - y_pred
    return np.sqrt(np.mean(error**2))

def r2F(y_true, y_pred):
    numerador = ((y_true - y_pred) ** 2).sum()
    denominador = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (numerador / denominador)
# Función para ajustar el modelo y evaluarlo
def ajustar_evaluar_modelo(X, y):
    coeficientes = regresion_manual(X, y)
    y_pred = predecir(X, coeficientes)
    r2_ = r2F(y, y_pred)
    rmse_val = rmse(y, y_pred)
    return coeficientes, y_pred, r2_, rmse_val

opcion=int(input())
# Cargar los datos
data = pd.read_csv(ruta)

# Definir las columnas de características (X) y la columna de objetivo (y)
if opcion == 1:
    print(len(data))
    print(len(data.columns))
    caracteristicas =['VTI_F','PEEP', 'BPM','VTE_F'] # Seleccionar características (variables independientes)
    objetivo = ['Pasos']  # Seleccionar variable objetivo (variable dependiente)
    print(caracteristicas)
    print(objetivo)
elif opcion == 2:
    X = data[['VTI_F']]
    y = data['Pasos']
    coef = regresion_manual(X, y)
    print(coef)
elif opcion == 3:
    X = data[['VTI_F']]
    y = data['Pasos']
    coef = regresion_manual(X, y)
    print(coef)
    y_pred = predecir(X,coef)
    r2_ = r2F(y, y_pred)
    rmse_val = rmse(y, y_pred)
     # imprimir los primeros 3 elementos de y e y_pred
    print(y[:3],  y_pred [:3])
    # imprimir r2 y rmse
    print(r2_,  rmse_val)
elif opcion==4:
    # modelo completo solo con VTI_F, completar la función ajustar_evaluar_modelo
    X_todo =data[['VTI_F']]
    y =data['Pasos']
    coeficientes_todo, y_pred_todo, r2_todo, rmse_todo = ajustar_evaluar_modelo(X_todo, y)
    print(r2_todo, rmse_todo)
elif opcion==5:
   # Completar la combinaciones de características de los modelos solicitados
    models = {
        'Modelo_1': ['VTI_F'],
        'Modelo_2': ['VTI_F', 'BPM'],
        'Modelo_3': ['VTI_F', 'PEEP'],
        'Modelo 4': ['VTI_F', 'PEEP', 'BPM'],
        'Modelo 5': ['VTI_F', 'PEEP', 'BPM', 'VTE_F']
      #COMPLETAR EL DICCIONARIO
             }
    for nombre_modelo, lista_caracteristicas in models.items():
        X = data[lista_caracteristicas] #data[completar]
        y = data['Pasos']
        coeficientes, y_pred, r2, rmse_val = ajustar_evaluar_modelo(X, y)
        print(nombre_modelo,r2, rmse_val)
        
elif opcion==6:
    # Modelos para cada combinación de PEEP y BPM
    valores_peep_unicos = data['PEEP'].unique()#completar sugerencia, utilizar unique()
    valores_bpm_unicos = data['BPM'].unique() #completar
    print(valores_peep_unicos)
    print(valores_bpm_unicos)
    predicciones_totales = []
    for peep in valores_peep_unicos:
        for bpm in valores_bpm_unicos:



            datos_subset = data[(data['PEEP'] == peep) & (data['BPM'] == bpm)] #completar el filtrado de datos, se deben filtrar los datos para cada para par de PEEP y BPM


            X_subset = datos_subset[['VTI_F']]
            y_subset = datos_subset['Pasos']
            coeficientes_subset, y_pred_subset, r2_subset, rmse_subset = ajustar_evaluar_modelo(X_subset, y_subset)
            print(peep, bpm, r2_subset, rmse_subset)
            predicciones_totales.append(y_pred_subset)
            
            # Graficar los valores 
            plt.scatter(X_subset, y_subset, label=f'PEEP: {peep:.2f}, BPM: {bpm:.2f}')

            # Graficar la recta de ajuste de la regresión lineal
            plt.plot(X_subset, X_subset * coeficientes_subset[1] + coeficientes_subset[0], color='red')

    # Agregar nombres a los ejes
    plt.xlabel('$VTI_F$', fontsize=18)
    plt.ylabel('$Pasos$', rotation=0, fontsize=18)

    plt.legend(['Recta de Ajuste'])

    plt.axis([200, 700, 16000, 32000])  
    for peep in valores_peep_unicos:
        for bpm in valores_bpm_unicos:
            plt.text(0.05, 0.95, f'PEEP: {peep:.2f}, BPM: {bpm:.2f}', ha='left', va='top', color='black', transform=plt.gca().transAxes, fontsize=10)

    plt.show()

    predicciones_concatenadas = np.concatenate(predicciones_totales)
    y=data['Pasos']
    r2_global = r2F(y, predicciones_concatenadas)
    rmse_global = rmse(y, predicciones_concatenadas)
    print('Global', r2_global, rmse_global)

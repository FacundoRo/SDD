# Distanciamiento Social con Vision Computacional
Proyecto realizado con python, deep learning and computer vision para monitorear el distanciamiento social.
Credito de idea: LandingAI

[![Mira el video](/images/sshot2.jpg)](https://drive.google.com/file/d/1AOnOwZZc6--YXz88jVEqRx66ecm4j1aD/view)
[![Mira el video](/images/sshot4.jpg)](https://drive.google.com/file/d/1AOnOwZZc6--YXz88jVEqRx66ecm4j1aD/view)

# Que es necesario para instalar
Es recomendable crear un nuevo entorno virtual para este proyecto e instalar las dependencias. Se pueden seguir los siguientes pasos para descargar comenzar con el proyecto

## Clone el repositorio
```
git clone https://github.com/FacundoRo/SDD.git
```
## Packages

Se necesita numpy y opencv


## Ejecutar el proyecto
```
cd SDD
python SDD_local.py
```
Al ejecutar SDD_local.py se abrirá una ventana del primer fotograma del video. En este punto, el código espera que el usuario marque 6 puntos haciendo clic en las posiciones apropiadas en el marco.

#### Primeros 4 puntos:
Los primeros 4 entre los 6 puntos requeridos se utilizan para marcar una region que va ser mapeada hacia un cuadrado. Además, las líneas marcadas por estos puntos deben ser líneas paralelas en el mundo real como se ve desde arriba. Por ejemplo, estas líneas podrían ser los bordes de la carretera. Estos 4 puntos deben proporcionarse en un orden predefinido que sigue.

* __Point1 (ai)__:  abajo a la izquierda
* __Point2 (ad)__: abajo a la derecha
* __Point3 (Ai)__: arriba a la izquierda
* __Point4 (Ad)__: arriba a la derecha


#### Últimos 2 puntos:
Los últimos dos puntos se usan para marcar dos puntos separados 2mts en la región de interés. Por ejemplo, esto podría ser la altura de una persona (más fácil de marcar en el marco)

El siguiente gif ayudará a comprender mejor los puntos
![mark4points](images/mark4points.gif)
![mark4points](images/pedestrian.gif)

## ¿Como funciona?
El diagrama de bloques completo del algoritmo se puede ver a continuación.
![Block Diagram](images/block_diagram.png)


## Advertencia

Ejecutar desde PC debido a que colab tiene algunas restricciones.
__Créditos de idea: LandingAI__

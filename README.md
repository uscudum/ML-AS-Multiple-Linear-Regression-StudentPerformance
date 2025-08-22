# Predicción de Rendimiento Estudiantil (Regresión Lineal)

Este proyecto muestra, paso a paso, cómo **explorar datos**,
**preprocesarlos** y **entrenar un modelo de Regresión Lineal** para
estimar el **índice de rendimiento académico** (`rendimiento`) a partir
de variables como **horas de estudio** y **horas de sueño**. Está
pensado para que estudiantes y docentes entiendan qué se hizo y puedan
**reproducir** el flujo completo.

------------------------------------------------------------------------

## Dataset

**Archivo**: `Student_Performance.csv`\
**Tamaño**: 10.000 filas, 6 columnas

Columnas originales (de `df.info()`): - `Hours Studied` *(int)*: Horas
dedicadas al estudio. - `Previous Scores` *(int)*: Calificaciones
anteriores. - `Extracurricular Activities` *(object)*: Participación en
actividades extracurriculares (**Yes/No**). - `Sleep Hours` *(int)*:
Horas promedio de sueño. - `Sample Question Papers Practiced` *(int)*:
Cantidad de guías/pruebas practicadas. - `Performance Index` *(float)*:
**Objetivo** -- índice de rendimiento.

Renombrado a español para facilitar la lectura en clase:

``` python
df.columns = ['horas', 'puntajePrevio', 'extra', 'suenio', 'pruebas', 'rendimiento']
```

Codificación de la variable categórica:

``` python
# Versión usada
df['extra'] = df['extra'].replace({'Yes': 1, 'No': 0})
```

------------------------------------------------------------------------

## Objetivo del ejercicio

Entrenar un **modelo de Regresión Lineal** para predecir `rendimiento`
usando como **features**: - `horas` (horas de estudio) - `suenio` (horas
de sueño)

Se estandarizan las variables para mejorar la estabilidad numérica del
ajuste y luego se **des-estandariza** la predicción para volver a la
escala original.

------------------------------------------------------------------------

## Análisis Exploratorio

1.  **Vista general**: `df.info()` y `df.head()` para tipos y primeras
    filas.

2.  **Relaciones bivariadas**:

    ``` python
    import seaborn as sns
    import matplotlib.pyplot as plt

    cols = ['horas', 'puntajePrevio', 'extra', 'suenio', 'pruebas', 'rendimiento']
    sns.pairplot(df[cols])
    plt.show()
    ```

3.  **Correlaciones**:

    ``` python
    sns.heatmap(df[cols].corr(), annot=True)
    plt.show()
    ```

> Propósito didáctico: observar tendencias lineales, posibles
> colinealidades y señales sobre qué variables podrían explicar mejor
> `rendimiento`.

------------------------------------------------------------------------

## Preprocesamiento

-   **Selección de variables**:

    ``` python
    X = df[['horas', 'suenio']].values     # independientes
    y = df[['rendimiento']].values         # dependiente (2D)
    ```

-   **Estandarización** (media 0, desvío 1):

    ``` python
    from sklearn.preprocessing import StandardScaler

    scalerX = StandardScaler()
    scalerY = StandardScaler()

    X_scaled = scalerX.fit_transform(X)
    y_scaled = scalerY.fit_transform(y)
    ```

> ℹ️ *Por qué estandarizar*: facilita la optimización y hace comparables
> las magnitudes.\
> *Nota*: en muchos casos no es necesario escalar `y`; aquí se hace con
> fin **pedagógico** y para mostrar la inversión de escala en la
> predicción.

------------------------------------------------------------------------

## Modelado

Entrenamiento de **Regresión Lineal**:

``` python
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_scaled, y_scaled)
```

------------------------------------------------------------------------

## Predicción (ejemplo con valores manuales)

``` python
import numpy as np

horasEstudio = 6
horasSuenio = 8

# 1) Escalamos la entrada
entradaEscalada = scalerX.transform(np.array([[horasEstudio, horasSuenio]]))

# 2) Predecimos en el espacio estandarizado
prediccionEscalada = modelo.predict(entradaEscalada)

# 3) Volvemos a la escala original
prediccion = scalerY.inverse_transform(prediccionEscalada)

print("El rendimiento estimado del estudiante es:", round(prediccion[0][0], 2), "%")
# Ejemplo de salida:
# El rendimiento estimado del estudiante es: 58.81 %
```

------------------------------------------------------------------------


## Licencia y uso

Este material es para fines **educativos**. Podés reutilizarlo en clases
citando la fuente del repositorio.

------------------------------------------------------------------------

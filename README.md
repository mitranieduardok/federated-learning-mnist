# Aprendizaje Federado con MNIST

Un flujo de trabajo de aprendizaje federado para clasificación de dígitos usando el dataset MNIST.
Cada integrante del equipo entrena un modelo local sobre su partición privada de datos, un servidor central
agrega las actualizaciones sin tener acceso a los datos crudos.

---

## Arquitectura - ResNet-Mini

```
Entrada (28×28×1)
  └─ Conv2D(32, 3×3) + BatchNorm + ReLU           ← Entrada inicial
  └─ BloquesResidual(32)                           ← Etapa 1  28×28×32
  └─ BloqueResidual(64, stride=2) → BloqueResidual(64)  ← Etapa 2  14×14×64
  └─ GlobalAveragePooling2D
  └─ Dropout(0.3)
  └─ Dense(10, softmax)
```

---

## Estructura del Repositorio

```
federated-learning-mnist/
├── TheModel.py              ← Arquitectura ResNet-Mini e interfaz de construcción
├── local_training.ipynb     ← Entrenamiento local, curvas de aprendizaje y reporte
├── global_aggregation.ipynb ← Comparación de FedAvg · FedMedian · FedAdam
└── README.md
```

> **`split_data.py`** se mantiene **fuera** de este repositorio (los datos son confidenciales).
> Divide MNIST en N particiones estratificadas y las guarda como archivos `.npz`.

---

## Flujo de Trabajo Federado

```
split_data.py  →  data_partitions/
                     client_0_data.npz   (Integrante 0)
                     client_1_data.npz   (Integrante 1)
                     client_2_data.npz   (Integrante 2)
                     client_3_data.npz   (Integrante 3)
                     client_4_data.npz   (Integrante 4)
                     client_5_data.npz   (Integrante 5)

Cada integrante corre local_training.ipynb  →  weights/client_N_weights.weights.h5

global_aggregation.ipynb agrega los pesos usando tres métodos:
  1. FedAvg      (McMahan et al., 2017)
  2. FedMedian   (Yin et al., 2018)
  3. FedAdam     (Reddi et al., 2021)
```

---

## Métodos de Agregación

### FedAvg
Promedio ponderado de los pesos de los clientes por tamaño de dataset — la línea base de la federación.

$$w_{\text{global}} = \sum_i \frac{n_i}{\sum_j n_j} w_i$$

### FedMedian
Mediana por coordenada en lugar de media.
Más robusto ante deriva de datos no-IID y clientes bizantinos (corruptos), ya que
la mediana no es afectada por valores extremos.

$$w_{\text{global}}[j] = \operatorname{mediana}(w_1[j], \ldots, w_N[j])$$

### FedAdam
Trata el delta de FedAvg como un pseudo-gradiente y aplica Adam del lado del servidor:

$$\Delta_t = \text{FedAvg}(\{w_i\}) - w_{\text{global}}^{(t)}$$
$$w^{(t+1)} = w^{(t)} + \eta_s \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Las tasas de aprendizaje adaptativas por parámetro típicamente aceleran la convergencia
en comparación con el paso fijo de FedAvg.

---

## Equipo

| Integrante | Client ID |
|------------|-----------|
| Alejandro  | 0 |
| Marcos     | 1 |
| Roberto    | 2 |
| Ivan       | 3 |
| Joaquin    | 4 |
| Eduardo    | 5 |

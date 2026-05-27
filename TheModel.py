"""
Aprendizaje Federado MNIST — Definición del Modelo Global
==========================================================
Arquitectura : ResNet-Mini

    Entrada(28×28×1)
      └─ Conv2D inicial(32, 3×3)  + BN + ReLU
      └─ BloqueResidual(32)
      └─ BloqueResidual(64, stride=2)   ← reduce dimensiones espaciales a la mitad
      └─ BloqueResidual(64)
      └─ GlobalAveragePooling2D
      └─ Dropout(0.3)
      └─ Dense(10, softmax)

Diferencias con la CNN base de clase (Sequential simple):
  • Conexiones residuales (skip connections) — permiten el flujo de gradientes en profundidad
  • BatchNormalization                        — estabiliza el entrenamiento y regulariza
  • GlobalAveragePooling2D                   — reemplaza Flatten, reduce parámetros
  • Sin capa densa oculta entre pooling y clasificador
  • Dropout en la cabeza del clasificador
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def _residual_block(x, filters: int, stride: int = 1):
    """Bloque residual de dos capas con shortcut proyectado si es necesario."""
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


class build:
    @staticmethod
    def build_it(
        input_shape: tuple = (28, 28, 1),
        num_classes: int = 10,
        learning_rate: float = 1e-3,
    ) -> tf.keras.Model:
        """Construye y compila el modelo ResNet-Mini para entrenamiento federado en MNIST.

        Parámetros
        ----------
        input_shape    : tupla HWC, por defecto (28, 28, 1)
        num_classes    : número de clases de salida, por defecto 10
        learning_rate  : tasa de aprendizaje de Adam, por defecto 1e-3

        Retorna
        -------
        tf.keras.Model compilado
        """
        inputs = Input(shape=input_shape)

        # Capa inicial
        x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Etapa 1 — 28×28×32
        x = _residual_block(x, 32)

        # Etapa 2 — 14×14×64
        x = _residual_block(x, 64, stride=2)
        x = _residual_block(x, 64)

        # Cabeza clasificadora
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = Model(inputs, outputs, name="ResNet_Mini")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


if __name__ == "__main__":
    m = build.build_it()
    m.summary()

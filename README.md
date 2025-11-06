# 🗑️ Waste Classification Using Transfer Learning & Fine-Tuning (IBM Project)

Este repositorio contiene el cuaderno **`Final Proj-Classify Waste Products Using TL FT.ipynb`**, realizado como parte del curso de **Deep Learning with Keras and Tensorflow IBM**.  
El objetivo del proyecto es **clasificar distintos tipos de residuos (plástico, papel, vidrio, metal, etc.)** mediante **Transfer Learning (TL)** y **Fine-Tuning (FT)** con redes neuronales convolucionales preentrenadas.

---

## 🎯 Objetivo
Aplicar técnicas de *Transfer Learning* sobre un modelo CNN preentrenado (como **MobileNetV2**, **ResNet50** o **VGG16**) para realizar la **clasificación automática de residuos** en imágenes, optimizando la precisión del modelo con Fine-Tuning en las últimas capas.

---

## 🗂️ Contenido del repositorio
- `Final Proj-Classify Waste Products Using TL FT.ipynb` → Notebook con el desarrollo completo del proyecto.
- `requirements.txt` → Dependencias necesarias para reproducir el entorno.
- `.gitignore` → Archivos ignorados en el control de versiones.

---

## 🧠 Metodología resumida

### 1️⃣ Carga y exploración de datos
- Dataset de imágenes de residuos clasificados por tipo.  
- División en carpetas: `/train`, `/validation`, `/test`.

### 2️⃣ Preprocesamiento
- Redimensionado de imágenes (`ImageDataGenerator` con `rescale=1./255`).
- *Data augmentation* para mejorar la generalización: rotación, zoom, flips horizontales.

### 3️⃣ Transfer Learning
- Carga de modelo preentrenado (`MobileNetV2` o `VGG16`) sin la última capa.
- Congelación de capas base (`base_model.trainable = False`).
- Adición de nuevas capas densas y de salida:
  ```python
  model = Sequential([
      base_model,
      GlobalAveragePooling2D(),
      Dense(128, activation='relu'),
      Dropout(0.3),
      Dense(num_classes, activation='softmax')
  ])

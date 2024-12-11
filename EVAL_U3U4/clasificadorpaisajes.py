import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Categorías de paisajes
CATEGORIES = ['Montañas', 'Desierto', 'Playa', 'Bosques']

# Función para preprocesar las imágenes
def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, str):  # Si es una ruta, carga la imagen
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    # Procesa la imagen como una matriz
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Cargar conjunto de datos de las carpetas
def load_dataset_from_folders(base_path, target_size=(64, 64)):
    images, labels = [], []
    for label, category in enumerate(CATEGORIES):
        folder_path = os.path.join(base_path, category)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            try:
                images.append(preprocess_image(image_path, target_size))
                labels.append(label)
            except Exception as e:
                print(f"Error al procesar la imagen {image_path}: {e}")
    return np.array(images), np.array(labels)

# Ruta base de las carpetas de entrenamiento
base_path = 'C:/Users/Admin/Documents/EVAL_U3U4'
X, y = load_dataset_from_folders(base_path)
y = to_categorical(y, num_classes=len(CATEGORIES))  # Codificación one-hot

# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(CATEGORIES), activation='softmax')  # Cuatro clases: montañas, desierto, playa, bosque
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Guardar el modelo entrenado
model.save('landscape_classifier_model.h5')

# Función para realizar predicción
def predict_landscape(image, model, target_size=(64, 64)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return CATEGORIES[class_idx]

# Función para capturar imagen desde la cámara y realizar predicción
def capture_and_predict(model):
    cap = cv2.VideoCapture(0)  # Iniciar la cámara
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    print("Presiona 'c' para capturar una imagen y realizar la predicción o 'q' para salir.")
    img_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el fotograma.")
            break

        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            prediction = predict_landscape(frame, model)
            print(f"Predicción: {prediction}")

            # Guardar la imagen capturada
            img_name = f"captured_image_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"Imagen guardada como {img_name}")
            img_counter += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Cargar el modelo entrenado
model = tf.keras.models.load_model('landscape_classifier_model.h5')

# Llamar a la función para capturar imágenes desde la cámara y realizar predicciones
capture_and_predict(model)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset =[
  ([37,2,3,4], [0, 0, 0, 1]),
  ([31,1,3,2], [1, 0, 0, 0]),
  ([27,1,3,2], [0, 1, 0, 0]),
  ([26,1,3,2], [0, 1, 0, 0]),
  ([26,1,3,1], [0, 1, 0, 0]),
  ([23,1,3,2], [1, 0, 0, 0]),
]

def create_model_ml():

  # Séparer les données et les étiquettes
  x_train = [item[0] for item in dataset]
  y_train = [item[1] for item in dataset]

  # Convertir en tableaux NumPy
  x_train = np.array(x_train)
  y_train = np.array(y_train)

  # Définition du modèle
  model = Sequential([
    # Première couche (couche cachée) avec 8 neurones et fonction d'activation ReLU
    Dense(12, activation='relu', input_shape=(4,)),  # 4 entrées
    # Couche de sortie avec 4 neurones, un pour chaque classe possible
    Dense(4, activation='softmax')  # Utilisation de softmax pour la classification
  ])

  # Compilation du modèle
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  # Affichage du résumé du modèle
  model.summary()

  # Entraînement du modèle
  model.fit(x_train, y_train, epochs=2000)

  # Evaluation du modèle
  loss, accuracy = model.evaluate(x_train, y_train)
  print(f"Loss: {loss}, Accuracy: {accuracy}")

  # Enregistrement du modèle
  model.save('/api/test_model.h5')  # sauvegarde le modèle en un fichier H5

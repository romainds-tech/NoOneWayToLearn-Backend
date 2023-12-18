import tensorflow as tf
import numpy as np


def predict_nowtl(age: float, cursus: float, side_project: float, open_source: float):
    # Charger le modèle
    model = tf.keras.models.load_model('/api/test_model.h5')

    # Injecter les données de prédiction
    data_de_prediction = np.array([[float(age), float(cursus), float(side_project), float(open_source)]])  # double crochet pour un batch de taille 1

    # Effectuer la prédiction
    predictions = model.predict(data_de_prediction)

    # Afficher la prédiction
    return np.round(predictions, 2)

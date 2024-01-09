import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

dataset = [
    ([99, 2, 3, 4], [0, 0, 0, 1]),
    ([37, 2, 3, 4], [0, 0, 0, 1]),
    ([31, 1, 3, 2], [1, 0, 0, 0]),
    ([27, 1, 3, 2], [0, 1, 0, 0]),
    ([26, 1, 3, 2], [0, 1, 0, 0]),
    ([26, 1, 3, 1], [0, 1, 0, 0]),
    ([23, 1, 3, 2], [1, 0, 0, 0]),
    ([12, 1, 3, 2], [1, 0, 0, 0]),
]


def normalize_dataset(dataset):
    # Extracting individual features
    ages = [one_line[0][0] for one_line in dataset]
    cursus = [one_line[0][1] for one_line in dataset]
    side_projects = [one_line[0][2] for one_line in dataset]
    open_sources = [one_line[0][3] for one_line in dataset]

    # Normalizing ages
    min_age = min(ages)
    max_age = max(ages)
    normalized_ages = [(age - min_age) / (max_age - min_age) for age in ages]

    # Normalizing cursus (binary feature)
    normalized_cursus = [(c - 1) for c in cursus]  # maps 1 to 0 and 2 to 1

    # Normalizing side_projects and open_source (1 to 4 range)
    normalized_side_projects = [(sp - 1) / 3 for sp in side_projects]
    normalized_open_sources = [(os - 1) / 3 for os in open_sources]

    # Reconstructing the dataset
    normalized_dataset = []
    for i in range(len(dataset)):
        normalized_features = [
            normalized_ages[i],
            normalized_cursus[i],
            normalized_side_projects[i],
            normalized_open_sources[i],
        ]
        normalized_dataset.append((normalized_features, dataset[i][1]))

    return normalized_dataset


def create_model_ml():
    dataset_normalized = normalize_dataset(dataset)

    # Séparer les données et les étiquettes
    x_train = [item[0] for item in dataset_normalized]
    y_train = [item[1] for item in dataset_normalized]

    # Convertir en tableaux NumPy
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Définition du modèle
    model = Sequential(
        [
            # Première couche (couche cachée) avec 12 neurones et fonction d'activation ReLU
            Dense(12, activation="relu", input_shape=(4,)),  # 4 entrées
            # Couche de sortie avec 4 neurones, un pour chaque classe possible
            Dense(
                4, activation="softmax"
            ),  # Utilisation de softmax pour la classification
        ]
    )

    # Compilation du modèle
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Affichage du résumé du modèle
    model.summary()

    # Entraînement du modèle
    model.fit(x_train, y_train, epochs=200)

    # Evaluation du modèle
    loss, accuracy = model.evaluate(x_train, y_train)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    # Enregistrement du modèle
    model.save("/api/test_model.h5")


def normalize_input(input_data, min_age, max_age):
    # Normalizing each feature
    normalized_age = (float(input_data[0]) - float(min_age)) / (
        float(max_age) - float(min_age)
    )
    normalized_cursus = float(input_data[1]) - 1  # maps 1 to 0 and 2 to 1
    normalized_side_project = (float(input_data[2]) - 1) / 3
    normalized_open_source = (float(input_data[3]) - 1) / 3

    return [
        normalized_age,
        normalized_cursus,
        normalized_side_project,
        normalized_open_source,
    ]

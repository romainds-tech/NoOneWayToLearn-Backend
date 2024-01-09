import asyncio
import os

import numpy as np
import tensorflow as tf
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from .models import IdeasList, Temoignage


def predict_nowtl(age: float, cursus: float, side_project: float, open_source: float):
    # Charger le modèle
    model = tf.keras.models.load_model("/api/test_model.h5")

    # Injecter les données de prédiction
    data_de_prediction = np.array(
        [[float(age), float(cursus), float(side_project), float(open_source)]]
    )  # double crochet pour un batch de taille 1

    # Effectuer la prédiction
    predictions = model.predict(data_de_prediction)

    # Afficher la prédiction
    return np.round(predictions, 2)


async def complet_one(word, parser, model, cursus, exp, temoi, appinf):
    print("start")
    prompt = PromptTemplate(
        template="Donne une liste d'idée en français de projet pertinent pour un développeur avec un cursus {cursus} et qui a {exp} ans d'expérience dans le monde professionel du développement. Son mode d'apprentissage informel le plus efficace est {appinf} .Ces idées de projet doivent faire environ 2 à 3 phrases et être en lien avec le mot clé : {word} et doivent également correspondre avec le témoignage de l'utilisateur concernant ce qu'il aime faire : {temoi} \n {format_instructions}\n",
        input_variables=["word", "cursus", "exp", "temoi", "appinf"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    resp = await chain.ainvoke(
        {"word": word, "cursus": cursus, "exp": exp, "temoi": temoi, "appinf": appinf}
    )
    print(word, " ok")

    return (word, resp)


async def generate_ideas_concurrently(
    key_words, parser, model, cursus, exp, temoi, appinf
):
    tasks = [
        complet_one(word, parser, model, cursus, exp, temoi, appinf)
        for word in key_words
    ]
    all_ideas = await asyncio.gather(*tasks)
    return all_ideas


async def process(
    cursus: str = "ingénieur",
    exp: str = "10",
    appinf: str = "Expérience professionnel",
    temoignage_query: str = "J'aime faire des poc de nouvelle techno, réaliser des side project. J'aime le python, le backend et la performance",
):
    os.environ.get("OPENAI_API_KEY")
    model = ChatOpenAI(temperature=0)
    parser = PydanticOutputParser(pydantic_object=Temoignage)

    print(temoignage_query)

    # Keywords extraction
    prompt = PromptTemplate(
        template="Donne une liste de mot clé pertinent extrait de ce témoignage utilisateur : \n{query} \n Renvoi les données exactement dans ce format : {format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    resp = chain.invoke({"query": temoignage_query})

    key_words = resp.key_words
    print(key_words)

    # ----- Ideas generation

    parser = PydanticOutputParser(pydantic_object=IdeasList)

    ideas = await generate_ideas_concurrently(
        key_words, parser, model, cursus, exp, temoignage_query, appinf
    )

    print([idea for tab in ideas for idea in tab[1].list_of_ideas])

    # Préparation des embeddings
    all_data = [idea for tab in ideas for idea in tab[1].list_of_ideas]

    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorized_data = embedder.embed_documents(all_data)

    assert len(all_data) == len(vectorized_data)
    print("all_data et vectorized_data font bien la même longueur")

    # Get five most similar vector
    temoignage_query_emb = embedder.embed_query(temoignage_query)

    similarities = cosine_similarity([temoignage_query_emb], vectorized_data)[0]

    sorted_indices = np.argsort(similarities)[::-1]
    bad_sorted_indices = np.argsort(similarities)

    top_matches = sorted_indices[:5]
    bad_matches = bad_sorted_indices[:5]

    for index in top_matches:
        print(f"Good simil : {all_data[index]}, Similarité: {similarities[index]}")

    for index in bad_matches:
        print(f"Bad simil: {all_data[index]}, Similarité: {similarities[index]}")

    return [all_data[index] for index in top_matches]

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field, validator


class Temoignage(BaseModel):
    temoignage: str = Field(description="Témoignage original")
    key_words: List[str] = Field(description="Liste des mots clés important du témoignage")

class IdeasList(BaseModel):
    list_of_ideas: List[str] = Field(description="List des idées pour l'utilisateur")
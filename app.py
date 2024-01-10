import datetime

from api.no_one_way_to_learn.ai.no_one_way_to_learn_model import (
    create_model_ml,
    normalize_input,
)
from api.no_one_way_to_learn.ai.predict import predict_nowtl, process
from api.no_one_way_to_learn.user.schemas import UserSchema
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(docs_url=None, redoc_url="/doc")
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


load_dotenv()


@app.post("/user", response_model=UserSchema, status_code=201)
@limiter.limit("120/minute")
async def index(request: Request, schema: UserSchema):
    return schema


@app.get("/create_model", status_code=200)
@limiter.limit("2/minute")
async def create_model(request: Request):
    create_model_ml()
    return {
        "message": "Model created successfully at : " + str(datetime.datetime.now())
    }


@app.get("/predict", status_code=200)
@limiter.limit("120/minute")
async def predict(request: Request, age, cursus, side_project, open_source):
    equivalent = {"never": "1", "sometimes": "2", "occasionally": "3", "lot": "4"}

    equi_cursus = {"tech": "1", "engineer": "2"}

    normalized_inputs = normalize_input(
        [age, equi_cursus[cursus], equivalent[side_project], equivalent[open_source]],
        12,
        99,
    )

    res = predict_nowtl(*normalized_inputs)

    print("res: ", res)

    return "Prediction successfully.", res.flatten().tolist()


@app.get("/generate_exercices", status_code=200)
@limiter.limit("2/minute")
async def generate_exercices(
    request: Request, cursus: str = "", exp: str = "", appinf: str = "", temoi: str = ""
):
    detail = await process(cursus, exp, appinf, temoi)
    return detail

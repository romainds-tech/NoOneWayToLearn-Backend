from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.no_one_way_to_learn.user.schemas import UserSchema
from api.no_one_way_to_learn.ai.no_one_way_to_learn_model import create_model_ml
from api.no_one_way_to_learn.ai.predict import predict_nowtl

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(docs_url=None, redoc_url="/doc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/user", response_model=UserSchema, status_code=201)
@limiter.limit("120/minute")
async def index(request: Request, schema: UserSchema):
    return schema


@app.get("/create_model", status_code=200)
@limiter.limit("1/minute")
async def create_model(request: Request):
    create_model_ml()
    return {"message": "Model created successfully."}


@app.get("/predict", status_code=200)
@limiter.limit("120/minute")
async def predict(request: Request):
    equivalent = {
        "never": "1",
        "sometimes": "2",
        "occasionally": "3",
        "lot": "4"
    }

    equi_cursus = {
        "tech": "1",
        "engineer": "2"
    }

    parameters = request.query_params

    print(parameters.items())

    age, cursus, side_project, open_source = parameters.items()

    res = predict_nowtl(age[1], equi_cursus[cursus[1]], equivalent[side_project[1]], equivalent[open_source[1]])

    print("res: ", res)

    return "Prediction successfully.", res.flatten().tolist()

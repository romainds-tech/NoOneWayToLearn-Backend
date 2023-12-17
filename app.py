from fastapi import FastAPI, Request
from api.no_one_way_to_learn.user.schemas import UserSchema

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)


app = FastAPI(docs_url=None, redoc_url="/doc")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/user", response_model=UserSchema, status_code=201)
@limiter.limit("120/minute")
async def index(request: Request, schema: UserSchema):
    return schema

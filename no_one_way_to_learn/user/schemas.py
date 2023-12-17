from pydantic import BaseModel, validator


class UserSchema(BaseModel):
    class Config:
        title = "User"

    username: str
    email: str
    full_name: str = None

    @validator("email")
    def validate_email(cls, email: str):
        if email.count("@") != 1 or len(email) < 1 or len(email) > 50:
            raise ValueError("This email address is not valid.")

        return email

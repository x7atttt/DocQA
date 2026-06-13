from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=128)


class UserInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    created_at: datetime | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import BizError
from app.core.response import ResponseCode, success_response
from app.core.security import create_access_token, hash_password, verify_password
from app.models import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserInfo

router = APIRouter()


@router.post("/register")
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    user = User(username=body.username, hashed_password=hash_password(body.password))
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise BizError(code=ResponseCode.USER_ALREADY_EXISTS, message="用户名已存在", http_status=409)
    await db.refresh(user)
    return success_response(UserInfo.model_validate(user).model_dump(mode="json"), "注册成功")


@router.post("/login")
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise BizError(
            code=ResponseCode.USERNAME_OR_PASSWORD_WRONG,
            message="用户名或密码错误",
            http_status=401,
        )
    token = create_access_token(user.id)
    data = TokenResponse(access_token=token, user=UserInfo.model_validate(user))
    return success_response(data.model_dump(mode="json"), "登录成功")

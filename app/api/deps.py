from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import AuthError
from app.core.response import ResponseCode
from app.core.security import decode_access_token
from app.models import User


async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError(code=ResponseCode.AUTH_FAILED, message="缺少认证信息")
    token = authorization.split(" ", 1)[1].strip()
    user_id = decode_access_token(token)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise AuthError(code=ResponseCode.AUTH_FAILED, message="用户不存在")
    return user

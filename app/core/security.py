from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from pwdlib import PasswordHash

from app.config import get_settings
from app.core.exceptions import AuthError
from app.core.response import ResponseCode

settings = get_settings()
pwd_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return pwd_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_hash.verify(plain_password, hashed_password)


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int:
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        sub = payload.get("sub")
        if sub is None:
            raise AuthError(code=ResponseCode.TOKEN_INVALID, message="无效的Token")
        return int(sub)
    except JWTError:
        raise AuthError(code=ResponseCode.TOKEN_INVALID, message="无效的Token")
    except (ValueError, TypeError):
        raise AuthError(code=ResponseCode.TOKEN_INVALID, message="无效的Token")

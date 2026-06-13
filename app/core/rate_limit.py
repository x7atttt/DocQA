from slowapi import Limiter
from slowapi.util import get_remote_address


def _user_or_ip_key(request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return f"user:{auth[7:].strip()[:32]}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_user_or_ip_key)

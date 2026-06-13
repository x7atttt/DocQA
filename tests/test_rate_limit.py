from unittest.mock import MagicMock

from app.core.rate_limit import _user_or_ip_key, limiter


def test_limiter_exists():
    assert limiter is not None
    assert callable(getattr(limiter, "_key_func", None))


def test_key_uses_bearer_token():
    req = MagicMock()
    req.headers = {"authorization": "Bearer abc123token"}
    key = _user_or_ip_key(req)
    assert key.startswith("user:")


def test_key_falls_back_to_ip_without_auth():
    req = MagicMock()
    req.headers = {}
    req.client = MagicMock(host="127.0.0.1")
    key = _user_or_ip_key(req)
    assert key.startswith("ip:")

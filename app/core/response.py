from typing import Any


class ResponseCode:
    SUCCESS = 0
    BAD_REQUEST = 40000
    VALIDATION_ERROR = 40001
    NOT_FOUND = 40004
    AUTH_FAILED = 10001
    TOKEN_EXPIRED = 10002
    TOKEN_INVALID = 10003
    USER_ALREADY_EXISTS = 10004
    USERNAME_OR_PASSWORD_WRONG = 10005

    DOC_NOT_FOUND = 20001
    DOC_PARSE_FAILED = 20002
    DOC_UPLOAD_FAILED = 20003
    UNSUPPORTED_FILE_TYPE = 20004

    CHAT_CREATE_FAILED = 30001
    EMPTY_QUESTION = 30002


def success_response(data: Any = None, message: str = "success") -> dict:
    return {"code": ResponseCode.SUCCESS, "message": message, "data": data}


def error_response(code: int, message: str) -> dict:
    return {"code": code, "message": message, "data": None}

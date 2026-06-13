from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.response import ResponseCode, error_response


class BizError(Exception):
    def __init__(self, code: int, message: str, http_status: int = 400):
        self.code = code
        self.message = message
        self.http_status = http_status
        super().__init__(message)


class AuthError(BizError):
    def __init__(self, code: int = ResponseCode.AUTH_FAILED, message: str = "认证失败", http_status: int = 401):
        super().__init__(code, message, http_status)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BizError)
    async def _biz_error_handler(_: Request, exc: BizError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=error_response(exc.code, exc.message))

    @app.exception_handler(AuthError)
    async def _auth_error_handler(_: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=error_response(exc.code, exc.message))

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "请求错误"
        code_map = {
            401: ResponseCode.AUTH_FAILED,
            404: ResponseCode.NOT_FOUND,
            405: ResponseCode.BAD_REQUEST,
            422: ResponseCode.VALIDATION_ERROR,
        }
        code = code_map.get(exc.status_code, ResponseCode.BAD_REQUEST)
        return JSONResponse(status_code=exc.status_code, content=error_response(code, message))

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_response(ResponseCode.VALIDATION_ERROR, "参数校验失败"),
        )

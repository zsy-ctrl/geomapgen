from __future__ import annotations

import sys
from typing import Callable, Optional


class GeoPipelineError(RuntimeError):
    def __init__(self, code: str, message: str, cause: Optional[BaseException] = None) -> None:
        self.code = str(code).strip().upper()
        self.message = str(message).strip()
        self.cause = cause
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def raise_geo_error(code: str, message: str, cause: Optional[BaseException] = None) -> None:
    err = GeoPipelineError(code=code, message=message, cause=cause)
    if cause is None:
        raise err
    raise err from cause


def wrap_geo_error(code: str, message: str, exc: BaseException) -> None:
    if isinstance(exc, GeoPipelineError):
        raise exc
    raise_geo_error(code=code, message=message, cause=exc)


def format_geo_exception(exc: BaseException, default_code: str) -> str:
    if isinstance(exc, GeoPipelineError):
        if exc.cause is not None:
            return (
                f"[{exc.code}] {exc.message} | "
                f"cause={type(exc.cause).__name__}: {exc.cause}"
            )
        return f"[{exc.code}] {exc.message}"
    return f"[{default_code}] {type(exc).__name__}: {exc}"


def run_with_geo_error_boundary(main_fn: Callable[[], None], default_code: str) -> None:
    try:
        main_fn()
    except SystemExit:
        raise
    except KeyboardInterrupt as exc:
        print(format_geo_exception(exc, default_code="GEO-0001"), file=sys.stderr, flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(format_geo_exception(exc, default_code=default_code), file=sys.stderr, flush=True)
        raise SystemExit(1)

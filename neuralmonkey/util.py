from typing import Any, Type
from typeguard import check_type


def match_type(value: Any, type_: Type) -> bool:
    try:
        check_type("value", value, type_, None)  # type: ignore
    except TypeError:
        return False
    return True

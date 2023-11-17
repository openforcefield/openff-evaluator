try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        HttpUrl,
        PositiveFloat,
        PositiveInt,
        ValidationError,
        confloat,
        conint,
        conlist,
        constr,
        root_validator,
        validator,
    )
    from pydantic.v1.validators import dict_validator
except ModuleNotFoundError:
    from pydantic import (
        BaseModel,
        Field,
        HttpUrl,
        PositiveFloat,
        PositiveInt,
        ValidationError,
        confloat,
        conint,
        conlist,
        constr,
        root_validator,
        validator,
    )
    from pydantic.validators import dict_validator

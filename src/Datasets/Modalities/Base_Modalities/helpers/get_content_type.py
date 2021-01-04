import ast
import numbers
import numpy as np

TYPE_UNKNOWN = 0x00
TYPE_NUMBER = 0x11
TYPE_NUMBER_AS_STR = 0x12
TYPE_BOOL = 0x21
TYPE_BOOL_AS_STR = 0x22
TYPE_STR = 0x31


def get_content_type(content):
    """
    Retrieve the content type of
    """
    cnt = 0
    content_type = None
    while content_type is None:
        t = content.iloc[cnt]
        cnt += 1
        if (cnt > 10):
            raise ValueError('Empty CSV column - first 10 values are missing blank')

        # check for empty types

        if isinstance(t, numbers.Number):
            return TYPE_NUMBER
        if isinstance(t, (bool, np.bool_)):
            return TYPE_BOOL

        if isinstance(t, str):
            try:
                t = ast.literal_eval(t)
                if isinstance(t, numbers.Number):
                    return TYPE_NUMBER_AS_STR
                elif isinstance(t, (bool, np.bool_)):
                    return TYPE_BOOL_AS_STR
            except Exception:
                return TYPE_STR

    if content_type is None:
        raise BaseException(f'Unknown csv content type: "{str(type(t))}"')

    return content_type

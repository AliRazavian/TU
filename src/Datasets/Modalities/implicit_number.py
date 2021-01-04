from .Base_Modalities.base_implicit import Base_Implicit
from .Base_Modalities.base_number import Base_Number


class Implicit_Number(Base_Implicit, Base_Number):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

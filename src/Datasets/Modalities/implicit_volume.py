from .Base_Modalities.base_implicit import Base_Implicit
from .Base_Modalities.base_volume import Base_Volume


class Implicit_Volume(Base_Implicit, Base_Volume):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

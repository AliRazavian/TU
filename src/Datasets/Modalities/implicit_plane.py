from .Base_Modalities.base_implicit import Base_Implicit
from .Base_Modalities.base_plane import Base_Plane


class Implicit_Plane(Base_Implicit, Base_Plane):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

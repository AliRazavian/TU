from .Base_Modalities.base_style import Base_Style
from .Base_Modalities.base_plane import Base_Plane


class Style_Plane(Base_Style, Base_Plane):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

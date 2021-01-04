from .Base_Modalities.base_style import Base_Style
from .Base_Modalities.base_volume import Base_Volume


class Style_Volume(Base_Style, Base_Volume):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

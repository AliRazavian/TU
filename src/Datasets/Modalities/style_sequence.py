from .Base_Modalities.base_style import Base_Style
from .Base_Modalities.base_sequence import Base_Sequence


class Style_Sequence(Base_Style, Base_Sequence):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

from .Base_Modalities.base_implicit import Base_Implicit
from .Base_Modalities.base_sequence import Base_Sequence


class Implicit_Sequence(Base_Implicit, Base_Sequence):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

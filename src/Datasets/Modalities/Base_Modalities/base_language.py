from abc import ABCMeta

from .base_sequence import Base_Sequence


class Base_Language(Base_Sequence, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.language_space = self.get_cfgs('language_space', default="en")
        self.case_sensitive = self.get_cfgs('case_sensitive', default=False)
        self.discard_numbers = self.get_cfgs('discard_numbers', default=True)
        self.sequence_length = self.get_cfgs('sentence_length', default=256)
        self.dictionary = self.get_cfgs('dictionary', default=' .,abcdefghijklmnopqrstuvwxyz$()')
        self.char_to_ix = {}
        for i in range(len(self.dictionary)):
            self.char_to_ix[self.dictionary[i]] = i

        self.set_channels(len(self.dictionary))
        self.set_width(self.sequence_length)

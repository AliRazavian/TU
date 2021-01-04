from GeneralHelpers import Singleton


class Base_Network_Factory(metaclass=Singleton):

    def __init__(self):
        # Protected storage of networks
        self._neural_nets = {}

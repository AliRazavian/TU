class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        elif len(args) > 0 or len(kwargs.keys()) > 0:
            raise ValueError('A singleton should only have values at initiation')

        return cls._instances[cls]

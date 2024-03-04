class Registry():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    def register(self, name, cls, parameters):
        self._registry[name] = (cls, parameters)

    def get(self, name):
        return self._registry.get(name)

    def make(self, name):
        if name not in self._registry:
            raise ValueError(f"Unknown environment: {name}")
        cls, parameters = self.get(name)
        return cls(**parameters)

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def __contains__(self, name):
        return name in self._registry

    def __getitem__(self, name):
        return self._registry[name]

    def __setitem__(self, name, value):
        self._registry[name] = value

    def __delitem__(self, name):
        del self._registry[name]

    def __repr__(self):
        return repr(self._registry)
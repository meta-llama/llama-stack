class Subcommand:
    """All llama cli subcommands must inherit from this class"""

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def _add_arguments(self):
        pass

from pytsbe.store.default_serializer import DefaultLibrarySerializer


class FedotSerializer(DefaultLibrarySerializer):
    """
    Serializer for FEDOT framework output. Can store predictions, serialize
    obtained pipelines, save composing history and pipelines pictures
    """

    def __init__(self, storage_paths: dict):
        super().__init__(storage_paths)


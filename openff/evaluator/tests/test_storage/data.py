from openff.evaluator.storage.attributes import QueryAttribute, StorageAttribute
from openff.evaluator.storage.data import BaseStoredData, HashableStoredData
from openff.evaluator.storage.query import BaseDataQuery


class SimpleData(BaseStoredData):
    some_attribute = StorageAttribute(docstring="", type_hint=int)

    @classmethod
    def has_ancillary_data(cls):
        return False

    def to_storage_query(self):
        return SimpleDataQuery.from_data_object(self)


class SimpleDataQuery(BaseDataQuery):
    @classmethod
    def data_class(cls):
        return SimpleData

    some_attribute = QueryAttribute(docstring="", type_hint=int)


class HashableData(HashableStoredData):
    some_attribute = StorageAttribute(docstring="", type_hint=int)

    @classmethod
    def has_ancillary_data(cls):
        return False

    def to_storage_query(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.some_attribute)

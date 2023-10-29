import re

class BaseProcessor(object):
    _processor_registry = {}

    @classmethod
    def register_processor(cls, processor_name):
        def decorator(subclass):
            cls._processor_registry[processor_name] = subclass
            return subclass
        return decorator

    def __new__(cls, processor_name, *args, **kwargs):
        if processor_name not in cls._processor_registry:
            raise ValueError(f"No processor registered with name '{processor_name}'")
        return super(BaseProcessor, cls).__new__(cls._processor_registry[processor_name])

    def set_model(self):
        raise NotImplementedError("The 'set_model' method must be implemented in derived classes.")

    def ask(self, query):
        information = self.ask_information(query)
        weight = self.ask_weight(information, query)
        return information, weight

    def ask_info(self, query, context, *args, **kwargs):
        raise NotImplementedError("The 'ask_information' method must be implemented in derived classes.")

    def ask_score(self, query, context, *args, **kwargs):
        raise NotImplementedError("The 'ask_weight' method must be implemented in derived classes.")
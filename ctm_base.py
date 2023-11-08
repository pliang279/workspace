from processors.processor_base import BaseProcessor
import concurrent.futures
import random

class BaseConsciousnessTuringMachine(object):
    _ctm_registry = {}

    @classmethod
    def register_ctm(cls, ctm_name):
        def decorator(subclass):
            cls._ctm_registry[ctm_name] = subclass
            return subclass
        return decorator

    def __new__(cls, ctm_name, *args, **kwargs):
        if ctm_name not in cls._ctm_registry:
            raise ValueError(f"No CTM registered with name '{ctm_name}'")
        return super(BaseConsciousnessTuringMachine, cls).__new__(cls._ctm_registry[ctm_name])

    def __init__(self):
        self.processor_list = []

    def add_processor(self, processor_name):
        processor_instance = BaseProcessor(processor_name)
        self.processor_list.append(processor_instance)

    def ask_processors(self, question):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            infos = list(executor.map(lambda processor: processor.ask_info(question), self.processor_list))
            scores = list(executor.map(lambda processor: processor.ask_score(question), self.processor_list))
        assert len(infos) == len(scores) == len(self.processor_list)
        return infos, scores 


    def uptree_sampling(self, infos, scores):
        max_score = max(scores)
        max_indices = [i for i, score in enumerate(scores) if score == max_score]
        index = random.choice(max_indices)
        
        return infos[index], scores[index]
    
    def downtree_broadcast(self, infos, scores, index):
        return infos, scores
        
    def link_form(self, infos, scores):
        return infos, scores
    
    def processor_fuse(self, infos, scores):
        return infos, scores
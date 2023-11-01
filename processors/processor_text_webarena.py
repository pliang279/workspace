from processor_base import BaseProcessor
import subprocess, os
import json


@BaseProcessor.register_processor('text_webarena')
class TextWebarenaProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()

    def init_processor(self):
        os.environ[
        "SHOPPING"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
        os.environ[
            "SHOPPING_ADMIN"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
        os.environ[
            "REDDIT"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
        os.environ[
            "GITLAB"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
        os.environ[
            "MAP"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
        os.environ[
            "WIKIPEDIA"
        ] = "http://192.168.0.109:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        os.environ[
            "HOMEPAGE"
        ] = "http://192.168.0.143:4399" 
        os.environ[
            "OPENAI_API_KEY"
        ] = "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"
    def process(self, payload: dict) -> str:
        with open('temp_output.txt', 'r+') as f:
            result = f.read().split('\n')
        return result[0]
    
    def score(self, payload: dict) -> str:
        with open('temp_output.txt', 'r+') as f:
            result = f.read().split('\n')
        return result[1]

    def ask_info(self, query: str, context:dict) -> str:
        # context_ to json
        json_object = json.dumps(context)
        with open('webarena/config_files/0.json', 'w+') as f:
            f.write(json_object)
        p = subprocess.call(
            ["python", "run.py", "--instruction_path", "agent/prompts/jsons/p_cot_id_actree_2s.json", "--test_start_idx", "0", "--test_end_idx", "1", "--model", "gpt-3.5-turbo", "--result_dir", "webarena_temp_output"],
            cwd='webarena'
        )
        answer: str = self.process(context)
        return answer
    
    def ask_score(self, query: str, context: str) -> float:
        json_object = json.dumps(context)
        with open('processors/webarena/config_files/0.json', 'w+') as f:
            f.write(json_object)
        p = subprocess.call(
            ["python", "run.py", "--instruction_path", "agent/prompts/jsons/p_cot_id_actree_2s.json", "--test_start_idx", "0", "--test_end_idx", "1", "--model", "gpt-3.5-turbo", "--result_dir", "webarena_temp_output"],
            cwd='webarena'
        )
        answer: str = self.score(context)
        return answer

if __name__ == "__main__":
    processor = BaseProcessor('text_webarena')
    question = "Tell me the full address of all international airports that are within a driving distance of 30 km to Carnegie Art Museum"
    request_dict = {'sites': ['map'], 'task_id': 9, 'require_login': True, 'storage_state': None, 'start_url': 'http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000', 'geolocation': None, 'intent_template': 'Tell me the full address of all {{airport_type}} that are within a driving distance of {{radius}} to {{start}}', 'instantiation_dict': {'airport_type': 'international airports', 'start': 'Carnegie Art Museum', 'radius': '30 km'}, 'intent': question, 'require_reset': False, 'eval': {'eval_types': ['string_match'], 'reference_answers': {'must_include': ['Pittsburgh International Airport, Southern Beltway, Findlay Township, Allegheny County, 15231, United States']}, 'reference_url': '', 'program_html': [], 'string_note': '', 'reference_answer_raw_annotation': 'Pittsburgh International Airport People Movers, Airport Boulevard, Findlay Township, Allegheny County, Pennsylvania, 15231, United States'}, 'intent_template_id': 79}
    summary: str = processor.ask_info(query=question, context=request_dict)
    print(summary)

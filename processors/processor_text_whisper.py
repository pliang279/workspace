from processor_base import BaseProcessor
import whisper
import math

@BaseProcessor.register_processor('text_whisper')
class TextWhisperProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()

    def init_processor(self):
        self.model_id: str = 'base'

    def process(self, payload: dict) -> dict:
        input = payload["inputs"]
        model = whisper.load_model(self.model_id)
        response = model.transcribe(input)
        return response

    def score(self, payload: dict) -> float:
        input_data: dict = payload
        output: dict = self.process(input_data)
        avg_score = 0
        for seg in output["segments"]:
            avg_score += math.exp(seg["avg_logprob"])
        result = avg_score/len(output["segments"])
        print(result)
        return result

    def ask_info(self, aud_file: str):
        input_data: dict = {"inputs": aud_file}
        output: dict = self.process(input_data)
        texts = []
        times = []
        for seg in output["segments"]:
            texts.append(seg["text"])
            times.append((seg["start"], seg["end"]))
        return (texts, times)

    def ask_score(self, aud_file: str) -> float:
        input_data: dict = {"inputs": aud_file}
        score: float = self.score(input_data)
        assert 0 <= score <= 1
        return score

if __name__ == "__main__":
    processor = BaseProcessor('text_whisper')
    aud_file: str = "test.mp3"
    output = processor.ask_info(aud_file)
    confidence: float = processor.ask_score(aud_file)
    print(f'Output: {output}, Confidence: {confidence}')
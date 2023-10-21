import openai
import re


class BaseProcessor(object):
    
    def __init__(self, processor_name):
        self.processor_name = processor_name

    def ask_information(self, question):
        """
        A generic ask method that can be overridden by derived classes.
        """
        raise NotImplementedError("The 'ask' method must be implemented in derived classes.")

    def process_response(self, response):
        """
        A method to process the response from OpenAI. Can be overridden for specific processing.
        """
        return response.choices[0].message['content'].strip()

    def extract_weight(self, response_content):
        """
        Extracts weight from a response using regex. Used for consistency across classes.
        """
        match = re.search(r'\b([1-5])\b', response_content)
        if match:
            return int(match.group(1))
        else:
            return None
        
    def ask(self, question):
        


class LanguageProcessor(BaseProcessor):
    def __init__(self, processor_name):
        super().__init__(processor_name)

    def ask_information(self, question):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.processor_name,
            messages=messages
        )
        
        return self.process_response(response)
    
    def ask_weight(self, information, question):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "what do you think '{}' is important for answering this question '{}'. Answering from 1 to 5. 1 represents very not related. 5 represents very related".format(information, question)}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.processor_name,
            messages=messages
        )
        
        return self.extract_weight(self.process_response(response))


class BaseConsciousnessTuringMachine(object):
    def __init__(self):
        self.processor_list = []

    def add_processor(self, processor_name):
        """Add a GPT processor with the given name."""
        processor = LanguageProcessor(processor_name)
        self.processor_list.append(processor)

    def ask_processors(self, question):
        """Send the question to all the processors and get the answers."""
        answers = [processor.ask(question) for processor in self.processor_list]
        return answers

    def output_processor_branish_language(self, answers):
        """Convert the answers to branish language (as an example)."""
        # This is a placeholder. You might have to implement the actual logic.
        branish_answers = ["branish:" + answer for answer in answers]
        return branish_answers
    
    def ask(self, question):


    def gain_feedback(self):
        return
    
    def output_processor_weight(self):
        return


class ConsciousnessTuringMachine(BaseConsciousnessTuringMachine):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    ctm = ConsciousnessTuringMachine()
    ctm.add_processor("gpt-3.5-turbo")
    ctm.add_processor("gpt-3.5-turbo")
    
    document1 = ['One man wearing white T-shirt is standing in a classroom', 'He is talking loudly about machine learning.']
    query = 'what is the name of its professor?'
    
    answers = ctm.ask_processors(query)
    print(answers)

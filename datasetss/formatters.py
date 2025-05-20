from utils_tokenization import SpecialTokens

class TokenizationFormatter(object):
    @classmethod
    def apply_formatting(cls, input_dict, **kwargs):
        return input_dict

class AcumenInstructionFormatterTrain(TokenizationFormatter):
    @classmethod
    def apply_formatting(cls, input_dict, **kwargs):
        if 'task_token' not in input_dict:
            print(input_dict)
            raise ValueError('task_token not found in input_dict')
        
        params = input_dict.get('param', '')
        if type(params) == tuple:
            params = ' '.join(params)
            
        task_description = f'{input_dict["task_token"]}{params}'
        
        return {
            'text': f'{task_description}{SpecialTokens.start_of_input}{input_dict["input"]}{SpecialTokens.end_of_input}{SpecialTokens.start_of_answer}{input_dict["answer"]}{SpecialTokens.end_of_answer}',
        }

class AcumenInstructionFormatterTest(TokenizationFormatter):
    @classmethod
    def apply_formatting(cls, input_dict, **kwargs):
        params = input_dict.get('param', '')
        if type(params) == tuple:
            params = ' '.join(params)
        
        task_description = f'{input_dict["task_token"]}{params}'

        return {
            'text': f'{task_description}{SpecialTokens.start_of_input}{input_dict["input"]}{SpecialTokens.end_of_input}{SpecialTokens.start_of_answer} ',
            'answer': f'{input_dict["answer"]}{SpecialTokens.end_of_answer}',
            'fulltext': f'{task_description}{SpecialTokens.start_of_input}{input_dict["input"]}{SpecialTokens.end_of_input}{SpecialTokens.start_of_answer}{input_dict["answer"]}{SpecialTokens.end_of_answer}',
        }

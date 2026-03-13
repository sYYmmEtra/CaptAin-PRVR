
from AutoSchema import AutoSchema
import json
import threading
import time

print_lock = threading.Lock()

class AutoPrompt:
    def __init__(self, prompt_template, output_schema=None, output_json_template=None, output_basemodel=None, is_output_json=True, model='lbot-7b', temperature=0, finetuned_model=None, is_debug=False, system_prompt = None, max_tokens = None, output_origin = False, guided_regex = None, function_call = None, enable_function_call = None, save_history = False):
        # 增加允许多轮输入的结构
        if isinstance(prompt_template, str):
            self.__prompt_templates = [prompt_template]

        elif isinstance(prompt_template, list):
            self.__prompt_templates = prompt_template

        else:
            raise ValueError('prompt_template must str or list, not{}'.format(type(prompt_template)))

        for i in range(len(self.__prompt_templates)):
            self.__prompt_templates[i] = self.__prompt_templates[i].strip()

        if output_json_template and output_basemodel:
            raise ValueError('Either json_template or BaseModel can be used.')
        
        output_schema = output_schema or output_json_template or output_basemodel
        
        if not output_schema:
            is_output_json = False

        if guided_regex != None and is_output_json:
            raise ValueError('guided_regex can not be used with is_output_json')

        if not function_call:
            enable_function_call = "none"
        else:
            if enable_function_call != "none":
                output_origin = True

        self.__multi_step_result = []
        self.__is_output_json = is_output_json
        self.__stream_output = False
        self.__is_debug = is_debug
        self.__output_schema = None
        self.__function_call_schema = None
        self.__enable_function_call = enable_function_call
        self.__system_prompt = system_prompt
        self.__output_origin = output_origin
        self.__guided_regex = guided_regex
        self.__save_history = save_history
        self.__history = []

        if output_schema:
            self.__output_schema = AutoSchema(output_schema)

        if self.__output_origin and (self.__stream_output or self.__is_output_json):
            raise ValueError('output_origin can not be used with stream_output or basemodel')


    def format(self, **args):
        format_output = []
        for prompt_template in self.__prompt_templates:
            input = prompt_template.format(**args)
            if self.__output_schema and prompt_template == self.__prompt_templates[-1]:
                schema_json = self.__output_schema.basemodel.schema_json(ensure_ascii=False)
                input += '\n```schema_json\n' + schema_json + '\n```'
            format_output.append(input)
        if len(format_output) == 1:
            return format_output[0]
        return format_output


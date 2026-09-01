from llm_sdk import Small_LLM_Model
from .parser import FunctonDefinition
from collections.abc import Generator
import numpy as np
import json
import sys
from enum import StrEnum

class Key(StrEnum):
    PROMPT = '"prompt":Ġ'
    NAME = ',Ġ"name":Ġ'
    PARAM = ',Ġ"parameters":Ġ'

class EngeneerTextFormat():
    """
    Engeneers the text parsed to the llm so that it may provide better logits
    as well as providing all function definitions.
    """
    def __init__(self, functions: list[FunctonDefinition]):
        """
        Sets needed class variables to format the prompt sent to the llm
        """
        self.all_funcs = functions
        self.user_prompt = "User prompt: {text}\n\n"

    def prompt_format(self, prompt: str) -> str:
        """
        Asks the llm to return the prompt in a valid json format
        """
        prompt_llm = ("I will provide you with a user prompt, and I want"
                     " it formatted as such: '\"prompt\": \"(user prompt)\",'"
                     "where user prompt is replaced with the prompt given")
        return prompt_llm + self.user_prompt.replace("text", prompt)

    def llm_parameters(self, func: FunctonDefinition) -> str:
        """
        Formats the paramaters so that they are readable and understandable.
        """
        if not func.parameters:
            return ""
        parameters = "({text})"
        variables = ""
        i = 0
        for var_name in func.parameters.keys():
            variables += var_name + ": " + func.parameters[var_name].type
            if i != len(func.parameters) - 1:
                variables += ", "
            i += 1
        return parameters.format(text=variables)

    def functions_format(self, prompt: str) -> str:
        """
        Formats the function information to send to the llm.
        """
        func_intro = ("I will provide you a prompt and a list of functions"
                      " with their name, description and parameters. From "
                      "this list of functions choose one that is most "
                      "suitable for the provided prompt, in this format:"
                      " ', \"name\": \"(function_name)\",'. If no function is"
                      " deemed suitable then return ', \"name\": \"None\"'.\n")
        func_exp: str = ""
        for func in self.all_funcs:
            func_exp += func.name + self.llm_parameters(func)
            func_exp += " -> " + func.returns.type + "\n"
            func_exp += func.description + "\n\n"
        return func_intro + func_exp + self.user_prompt.replace("text", prompt)

    def params_format(self, prompt: str, func: FunctonDefinition) -> str:
        """
        Asks the llm to return the appropriate parameters.
        """
        llm_request = ("Please use the prompt and the function definition"
                       " and parameters to make a formated json paramater"
                       " as shown below:"
                        "', \"parameters\": {\"(parameter key)\":"
                        " \"(parameter value)\",...}'")
        function_des = (func.name + self.llm_parameters(func) + "\n"
                        + func.description)
        return llm_request + self.user_prompt.replace("text", prompt) + function_des


class ConstrainedDecoding():
    """
    Reduces the probably number of logits that can be chosen by determining
    what should come next with the current llm output.
    """
    def __init__(self, token_dictionary: dict[str, dict[str, int]],
                 functions: list[FunctonDefinition]) -> None:
        self.functions: list[FunctonDefinition] = functions
        self.token_dict = token_dictionary
        self.prompt: str = ""
        self.output: str = ""
        self.chosen_func: (FunctonDefinition | None) = None
        self.prediction: str = "{"
        self.prdedicition_construction()

    def prdedicition_construction(self) -> None:
        self.prediction += f'{Key.PROMPT}<name>'
        self.prediction += f'{Key.NAME}<function>'
        self.prediction += f'{Key.PARAM}<parameters>' + '}'
        self.prediction = self.prediction.replace(" ", "Ġ")
        print(self.prediction)

    def update_prompt(self, prompt: str) -> None:
        print("updating prompt")
        self.prompt = '"' + prompt.replace(" ", "Ġ") + '"'
        self.prediction = self.prediction.replace("<name>", self.prompt)
        print(self.prediction)

    def update_paramaters(self, func: FunctonDefinition) -> None:
        parameter = func.parameters
        param_str = "{"
        index = 0
        for variable in parameter.keys():
            param_str += '"' + variable + '": '
            param_str += "<" + parameter[variable].type + ">"
            index += 1
            if index != len(parameter.keys()):
                param_str += ", "
        param_str += "}"
        self.prediction = self.prediction.replace("<parameter>", param_str)

    def set_invalids_to_infinity(self, logits: list[float],
                                 valid_token_ids: list[int]) -> list[float]:
        useful_logits: list[float] = []
        for token_id in valid_token_ids:
            useful_logits.append(logits[token_id])
        corrected_logits = [float("-inf")] * len(logits)
        index = 0
        for token_id in valid_token_ids:
            corrected_logits[token_id] = useful_logits[index]
            index += 1
        return corrected_logits

    def convert_token_to_id(self, target_list: list[str]) -> list[int]:
        useful_token_ids: list[int] = []
        for target in target_list:
            useful_token_ids.append(self.token_dict[target[0]][target])
        if len(useful_token_ids) == 0:
            raise ValueError("Error: No valid tokens were found.")
        print("token_ids:", useful_token_ids)
        return useful_token_ids

    def find_diff_in_words(self, target_word: str, string: str) -> str:
        partially_constructed = ""
        for letter in target_word:
            partially_constructed += letter
            if string.find(partially_constructed) != -1:
                continue
            return letter
        return ""

    def check_func_name(self, output: str, func: (str | None) = None) -> bool:
        if func is None:
            print("Searching output for func name")
            for single_func in self.functions:
                if output.find('"' + single_func.name + '"') != -1:
                    return True
            return False
        if Key.NAME in output:
            print("Updating output")
            output = output[output.find(Key.NAME) + len(Key.NAME):]
            output.remove(",")
        print("checking diff between", output, "vs", func)
        print(self.find_diff_in_words(output, func))
        if not self.find_diff_in_words(output, func):
            return True
        return False

    def find_type(self, value: str, ch_type: type) -> bool:
        if ch_type == "<number>":
            func = float
        elif ch_type == "<string>":
            func = str
        elif ch_type == "<boolean>":
            func = bool
        else:
            func = int
        try:
            func(value)
        except ValueError:
            return False
        return True

    def find_valid_function_token_ids(self, output: str) -> list[int]:
        valid_tokens: list[str] = []
        valid_funcs: list[str] = []
        print(output)
        if Key.NAME not in output:
            print("Not name in output leaving")
            return
        output = output[output.find(Key.NAME) + len(Key.NAME):]
        print("output is:", output)
        for func in self.functions:
            func_name = '"' + func.name + '"'
            if self.check_func_name(output, func_name) is True:
                print("adding a function")
                valid_funcs.append(func_name)
        print("valid funcs", valid_funcs)
        for func in valid_funcs:
            bucket = self.find_diff_in_words(func, output)
            for token in self.token_dict[bucket]:
                if not self.find_diff_in_words(output + token, func):
                    print("adding token", token)
                    valid_tokens.append(token)
        return self.convert_token_to_id(valid_tokens)
    
    def find_valid_token_ids_in_bucket(self, bucket: str,
                                       compare: str) -> list[int]:
        valid_tokens : list[int] = []
        print("Bucket still is", bucket)
        print(self.output)
        if bucket not in self.token_dict.keys():
            raise KeyError(f"Error: No '{bucket}' key in token_dictionary")
        for token in self.token_dict[bucket]:
            if compare.find(self.output + token) != -1:
                valid_tokens.append(token)
        print("valid:")
        print([token for token in valid_tokens])
        return self.convert_token_to_id(valid_tokens)

    def find_stage(self, output: str) -> tuple[str]:
        if Key.PROMPT not in output:
            self.output = output
            print("Could not find prompt")
            return (self.find_diff_in_words(Key.PROMPT, output), Key.PROMPT)
        elif self.prompt not in output:
            self.output = output[output.find(Key.PROMPT) + len(Key.PROMPT):]
            if self.output.count('"') < 2 and len(self.output) > len(self.prompt):
                raise ValueError("Error: LLM could not produce the right "
                                 "prompt")
            print("could not find prompt value")
            return (self.find_diff_in_words(self.prompt, self.output), self.prompt)
        elif Key.NAME not in output:
            print("Could not find name")
            print(self.output)
            self.output = output[output.find(self.prompt) + len(self.prompt):]
            print("Output is: ", self.output)
            return (self.find_diff_in_words(Key.NAME, self.output), Key.NAME)
        elif Key.NAME in output and self.chosen_func is None:
            print("checking for func names")
            self.output = output[output.find(Key.NAME) + len(Key.NAME):]
            if not self.check_func_name(self.output):
                return ("", "<Function>")
            for func in self.functions:
                if self.check_func_name(self.output, '"' + func.name + '"'):
                    print("found func", func.name)
                    self.chosen_func = func
        if Key.PARAM not in output:
            print("Searching for Parameter")
            if self.chosen_func is None:
                raise ValueError("Error: Could not find function")
            chosen_func_name = '"' + self.chosen_func.name + '"'
            self.output = output[output.find(chosen_func_name)
                                 + len(chosen_func_name):]
            return (self.find_diff_in_words(Key.PARAM, self.output), Key.PARAM)
        else:
            print("Searching for paramas")

    def correct_logits(self, logits: list[float], cur_output: str) -> (list[float] | None):
        print("\nNew logits:")
        print(self.prediction)
        cur_output = cur_output.replace(" ", "Ġ")
        bucket, compare = self.find_stage(cur_output)
        if compare == "<Function>":
            valid_token_ids = self.find_valid_function_token_ids(cur_output)
        else:
            valid_token_ids = self.find_valid_token_ids_in_bucket(bucket, compare)
        if valid_token_ids is None:
            return None
        return self.set_invalids_to_infinity(logits, valid_token_ids)


class LLMProcessing():
    """
    Class that handles the llm processes such as encoding, fetching logits,
    token selection and decoding.
    """
    def __init__(self, prompts: list[str], functions: list[FunctonDefinition]):
        """
        Initialises the llm model, aswell as setting up the llm prompts, along
        with other necessary variables.
        """
        self.llm = Small_LLM_Model()
        self.prompts = prompts
        self.functions = functions
        self.token_dictionary: dict[str, dict[str, int]] = {}
        self.create_token_to_token_id_dict()
        self.const_decode = ConstrainedDecoding(self.token_dictionary,
                                                functions)
        self.eng_text = EngeneerTextFormat(functions)
        self.encoded: list[int] = []
        self.output: str = ""
        self.cur_function : (FunctonDefinition | None) = None

    def create_token_to_token_id_dict(self) -> None:
        """
        Creates a dictionary to allow efficient look up of token id to token
        """
        with open(self.llm.get_path_to_vocab_file()) as f:
            token_to_token_id = json.load(f)
        for token, token_id in token_to_token_id.items():
            if token[0] not in self.token_dictionary.keys():
                self.token_dictionary[token[0]] = {}
            self.token_dictionary[token[0]][token] = token_id

    def encode_text_gen(self, prompt) -> None:
        """
        Encodes the engeneered text into token ids for the llm to process.
        """
        request: str = ""
        if (Key.PROMPT.replace("Ġ", " ") not in self.output
           or prompt.replace("Ġ", " ") not in self.output):
            print("prompt request:")
            request = self.eng_text.prompt_format(prompt)
        elif (Key.NAME.replace("Ġ", " ") not in self.output
             or not self.const_decode.check_func_name(self.output)):
              print("Function request:")
              request = self.eng_text.functions_format(prompt)
        else:
            print(self.output)
            for func in self.functions:
                func_name = '"' + func.name + '"'
                output = self.output.replace(" ", "Ġ")
                if self.const_decode.check_func_name(output, func_name):
                    self.cur_function = func
            print("Params request")
            if self.cur_function is None:
                raise ValueError("Error: Could not find the correct function")
            request = self.eng_text.params_format(prompt, self.cur_function)
        if request:
            self.encoded = self.llm.encode(request).tolist()[0]
            self.encoded += self.llm.encode(self.output).tolist()[0]

    def token_selection(self) -> float:
        """
        Chooses best token based off llm's probability and constrained decoding
        """
        logits = self.llm.get_logits_from_input_ids(self.encoded)
        if len(logits) == 0:
            raise ValueError("Error: No tokens were found")
        print(f"current output '{self.output}'")
        cor_logits = self.const_decode.correct_logits(logits, self.output)
        if cor_logits is None:
            return -1
        best_token_id = int(np.argmax(cor_logits))
        return best_token_id

    def prompt_process(self, prompt: str) -> None:
        """
        For each prompt in file it sends the prompt to the necessary functions
        so that it may be encoded, tokenised, logitised and produce the
        desired json output for each prompt to write to the output file.
        """
        self.output = ""
        i = 0
        while True:
            if not self.output or "," in self.llm.decode(next_token_id):
                self.encode_text_gen(prompt)
            print("Choosing next token round:", i)
            next_token_id = self.token_selection()
            if next_token_id == -1:
                print("End of llm")
                break
            print(f"LLM has chosen: '{self.llm.decode(next_token_id)}'")
            self.encoded.append(next_token_id)
            self.output += self.llm.decode(next_token_id)
            i += 1
            if i == 40:
                break
        print("\nLLM response:")
        print(f"'{self.output}'")

    def all_prompt_process(self) -> None:
        print("\nProcessing all prompts")
        for prompt in self.prompts:
            self.const_decode.update_prompt(prompt)
            self.prompt_process(prompt)
            break

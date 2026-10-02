from llm_sdk import Small_LLM_Model
from .parser import FunctonDefinition
from collections.abc import Generator
import numpy as np
import json
import sys
from enum import StrEnum

class Variable():
    index: int = 0
    var_index: int = -1
    var: str = ""
    type: str = ""

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
                     "where user prompt is replaced with the prompt given.\n")
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
        llm_request = ("Extract only the parameter values required by the function.\n"
                       "Do not include command words such as greet or reverse.\n"
                       "Return only this format:\n"
                       "', \"parameters\": {\"parameter\": value}'\n"
                       "Examples:\n"
                       "Prompt: greet Shrek\n"
                       "Output: ', \"parameters\": {\"name\": \"Shrek\"}'\n")
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
        self.makes()

    def makes(self) -> None:
        """
        makes resetabble vars for Constrained Decoding
        """
        self.prompt: str = ""
        self.chosen_func: (FunctonDefinition | None) = None
        self.cur_var: Variable = Variable()
        self.output: str = ""

    def update_prompt(self, prompt: (str | None) = None) -> None:
        if prompt:
            print("updating prompt")
            self.prompt = json.dumps(prompt).replace(" ", "Ġ")
            print("Updated prompt:", self.prompt)
        mode = False
        if self.chosen_func:
            print("Found chosen func")
            for var in self.chosen_func.parameters.keys():
                if self.chosen_func.parameters[var].type == "number":
                    mode = True
        if mode is True:
            print("Found numbers")
            numbers: list[str] = []
            number = ""
            for index in range(0, len(self.prompt)):
                try:
                    if self.prompt[index] != ".":
                        int(self.prompt[index])
                    number += self.prompt[index]
                except:
                    if number and "." not in number:
                        numbers.append(number)
                        number = ""
                        index += 2
                    continue
            for number in numbers:
                print(number, number + ".0")
                self.prompt = self.prompt.replace(number, number + ".0")
            print(self.prompt)
            print("Updated to account number for prompt", self.prompt)

    def update_current_var(self, output: str) -> bool:
        print("updating cur var")
        self.cur_var.var_index += 1
        all_params = list(self.chosen_func.parameters.keys())
        if self.cur_var.var_index >= len(all_params):
            print("Failed")
            return False
        self.cur_var.index = len(output)
        if self.cur_var.var_index != 0:
            self.cur_var.index -= 1
        new_var = all_params[self.cur_var.var_index]
        self.cur_var.var = '"' + new_var + '":Ġ'
        if self.cur_var.var_index != 0:
            self.cur_var.var = "Ġ" + self.cur_var.var
        self.cur_var.type = self.chosen_func.parameters[new_var].type
        print("succeded")
        return True

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
        print("checking diff between", output, "vs", func)
        print(self.find_diff_in_words(output, func))
        if not self.find_diff_in_words(output, func):
            return True
        return False

    def check_var_value(self, output: str) -> bool:
        print("Checking var value")
        print(output, "vs", self.prompt)
        output = output[len(self.cur_var.var):]
        print(f"Searching for finished value in '{output}'", self.cur_var.index)
        if output == "":
            return False
        if self.cur_var.type == "string":
            if output.count('"') < 2:
                return False
            index = output.rfind('"')
            mode = True
            while True:
                cur_character = output[index - 1]
                if cur_character == "\\":
                    if mode is True:
                        mode = False
                    else:
                        mode = True
                    index -= 1
                    if index == 0:
                        break
                else:
                    break
            return mode
        elif self.cur_var.type == "integer":
            index = self.prompt.find(output)
            if index == len(self.prompt) - 1:
                return True
            if index == -1:
                raise ValueError("Error: Parameter is the wrong value")
            try:
                int(self.prompt[index + len(output)])
                return False
            except ValueError:
                return True
        elif self.cur_var.type == "number":
            if "." not in output or output[-1] == ".":
                return False
            index = self.prompt.find(output) + len(output)
            print("Found value in", self.prompt[index:])
            if index == len(output) - 1:
                print("At the end of output all true")
                return True
            if index == -1:
                raise ValueError("Error: Parameter is the wrong value")
            try:
                int(self.prompt[index + 1])
                print(f"'{self.prompt[index + 1]}' is an int")
                print("more ints to be found")
                return False
            except (ValueError, IndexError):
                print("found all ints")
                return True
        elif self.cur_var.type == "boolean":
            if output == "true" or output == "false":
                return True
            return False
        return False

    def find_params_value_buckets(self, output: str) -> list[int]:
        valid_tokens: list[str] = []
        func = None
        if self.cur_var.type == "string":
            if len(output) == 0:
                return self.convert_token_to_id(['"'])
            elif len(output) > 1:
                print(output)
                valid_tokens.append('"')
            print(output)
            output = output[1:]
        for bucket in self.token_dict.keys():
            try:
                if self.cur_var.type == "integer":
                    int(bucket)
                    func = int
                elif self.cur_var.type == "number":
                    if bucket != ".":
                        float(bucket)
                    func = float
                elif self.cur_var.type == "boolean":
                    if bucket != "t" or bucket != "f":
                        raise ValueError
            except ValueError:
                 continue
            for token in self.token_dict[bucket].keys():
                if len(output) == 0 and token[0] == "Ġ":
                    continue
                try:
                    if not func and self.cur_var.type == "boolean":
                        if token == "true" or token == "false":
                            valid_tokens.append(token)
                    elif self.cur_var.type != "string":
                        if token[0] == "." and "." not in output:
                            valid_tokens.append(token)
                            continue
                        func(token)
                        print(token)
                    if output + token in self.prompt:
                        valid_tokens.append(token)
                except:
                    continue
        return self.convert_token_to_id(valid_tokens)

    def find_param_stage(self, output: str) -> list[int]:
        print("Finding params stage")
        output = output[output.find(Key.PARAM) + len(Key.PARAM):]
        print(f"New output: '{output}'")
        if not self.cur_var.var:
            print("didnt find var")
            if self.update_current_var(output) == False:
                print("Cant update")
                return []
            else:
                print(self.cur_var.var)
        if not output:
            print("Nothing in output")
            return self.convert_token_to_id(["{"])
        else:
            print("something in output")
        print("Passing starting stage")
        if "{" == output[0]:
            output = output[1:]
        if self.cur_var.var_index != 0:
            output = output[self.cur_var.index:]
            print(f"current output is '{output}'")
        diff = self.find_diff_in_words(self.cur_var.var, output)
        if diff:
            print("Found difference")
            return self.convert_token_to_id([diff])
        elif not self.check_var_value(output):
            print("Finding param Value")
            output = output[self.cur_var.index:]
            output = output[len(self.cur_var.var):]
            print("Sending in", output)
            return self.find_params_value_buckets(output)
        else:
            print("Value is complete")
            self.cur_var.var = None
            if len(self.chosen_func.parameters) == self.cur_var.var_index + 1:
                return self.convert_token_to_id(["}"])
            print(len(self.chosen_func.parameters), self.cur_var.var_index + 1)
            return self.convert_token_to_id([","])

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
            if compare.startswith(self.output + token):
                valid_tokens.append(token)
        print("valid:")
        print([token for token in valid_tokens])
        return self.convert_token_to_id(valid_tokens)

    def find_stage(self, output: str) -> tuple[str]:
        if Key.PROMPT not in output:
            self.output = output
            print("Could not find prompt")
            return (self.find_diff_in_words(Key.PROMPT, output), Key.PROMPT)
        elif self.prompt not in output and Key.NAME not in output:
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
                print("Couldnt find function")
                return ("", "<Function>")
            for func in self.functions:
                if self.check_func_name(self.output, '"' + func.name + '"'):
                    print("found func", func.name)
                    self.chosen_func = func
                    self.update_prompt()
                    break
        if Key.PARAM not in output:
            print(self.chosen_func)
            print("Searching for Parameter")
            if self.chosen_func is None:
                raise ValueError("Error: Could not find function")
            chosen_func_name = '"' + self.chosen_func.name + '"'
            self.output = output[output.find(chosen_func_name)
                                 + len(chosen_func_name):]
            return (self.find_diff_in_words(Key.PARAM, self.output), Key.PARAM)
        else:
            return("", "<Parameter>")

    def correct_logits(self, logits: list[float], cur_output: str) -> (list[float] | None):
        print("\nNew logits:")
        cur_output = cur_output.replace(" ", "Ġ")
        bucket, compare = self.find_stage(cur_output)
        if compare == "<Function>":
            valid_token_ids = self.find_valid_function_token_ids(cur_output)
        elif compare == "<Parameter>":
            valid_token_ids = self.find_param_stage(cur_output)
        else:
            valid_token_ids = self.find_valid_token_ids_in_bucket(bucket, compare)
        if not valid_token_ids:
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
        self.mode = "prompt"

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
              self.mode = "function"
        else:
            print("Sending ")
            print(self.output)
            cur_func: (FunctonDefinition | None) = None
            for func in self.functions:
                func_name = '"' + func.name + '"'
                output = self.output.replace(" ", "Ġ")
                output = output[: -1]
                if self.const_decode.check_func_name(output, func_name):
                    cur_function = func
            print("Params request")
            if cur_function is None:
                raise ValueError("Error: Could not find the correct function")
            request = self.eng_text.params_format(prompt, cur_function)
            self.mode = "param"
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
        self.mode = "prompt"
        self.const_decode.chosen_func = None
        self.const_decode.params = ""
        while True:
            if (not self.output or "," in self.llm.decode(next_token_id)
               and self.mode != "param"):
                self.encode_text_gen(prompt)
                if self.output and "," in self.llm.decode(next_token_id):
                    print(self.llm.decode(next_token_id))
            next_token_id = self.token_selection()
            if next_token_id == -1:
                print("End of llm")
                break
            print(f"LLM has chosen: '{self.llm.decode(next_token_id)}'")
            self.encoded.append(next_token_id)
            self.output += self.llm.decode(next_token_id)
            print("End output:", self.output)
            if (Key.PARAM.replace("Ġ", " ") in self.output
               and self.output.find("{") != -1 and self.output.find("}") != -1):
                break
        print("\nLLM response:")
        print(f"'{self.output}'")

    def all_prompt_process(self) -> None:
        print("\nProcessing all prompts")
        index = 0
        for prompt in self.prompts[8:]:
            print(len(self.prompts), "vs", index)
            print("\n\n\nNEW PROMPT!!!!!\n\n\n", prompt)
            print("User Prompt:", prompt)
            self.const_decode.makes()
            self.const_decode.update_prompt(prompt)
            self.prompt_process(prompt)
            index += 1

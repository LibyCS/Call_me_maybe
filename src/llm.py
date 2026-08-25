from llm_sdk import Small_LLM_Model
from .parser import FunctonDefinition
from collections.abc import Generator
import numpy as np
import json
import sys
from enum import StrEnum

class Key(StrEnum):
    PROMPT = '"prompt":'
    NAME = '"name":'
    FUNCS = 'function_name'
    PARAM = '"parameters":'

class EngeneerTextFormat():
    """
    Engeneers the text parsed to the llm so that it may provide better logits
    as well as providing all function definitions.
    """
    def __init__(self, functions: list[FunctonDefinition]):
        """
        Sets needed class variables to format the prompt sent to the llm
        """
        self.intro = ("I want you to choose the appropriate function based off"
                      " the user prompt from a set of functions and return it "
                      "in json format, I will provide the function's name, "
                      "paramaters and description.\n\n")
        self.all_funcs = functions
        self.func_exp = ""
        self.functions_format()
        self.user_prompt = "User prompt: {text}\n\n"
        self.format = ("Please return your answer in json format "
                       "formated as:\n\"prompt\": (user prompt),\n"
                       "\"name\": (function name),\n\"parameters\""
                       ": {(parameter key): (parameter value)...} "
                       "and nothing else.")

    def parameters_format(self, func: FunctonDefinition) -> str:
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

    def functions_format(self) -> None:
        """
        Formats the function information to send to the llm.
        """
        for func in self.all_funcs:
            self.func_exp += func.name + self.parameters_format(func)
            self.func_exp += " -> " + func.returns.type + "\n"
            self.func_exp += func.description + "\n\n"

    def create_llm_prompt(self, prompt: str) -> str:
        """
        Adds all neccessary information into one string so that it is now
        ready to send to the llm.
        """
        user = self.user_prompt.format(text=prompt)
        llm_prompt = (self.intro + self.func_exp + user + self.format)
        return llm_prompt


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
        self.prediction: str = "{"
        self.prdedicition_construction()

    def prdedicition_construction(self) -> None:
        self.prediction += f'{Key.PROMPT} <name>' + ','
        self.prediction += f'{Key.NAME} <function>' + ','
        self.prediction += f'{Key.PARAM} <parameters>' + '}'
        self.prediction = self.prediction.replace(" ", "Ġ")
        print(self.prediction)

    def update_prompt(self, prompt: str) -> None:
        print("updating prompt")
        self.prompt = prompt.replace(" ", "Ġ")
        self.prediction = self.prediction.replace("<name>", self.prompt)
        print(self.prediction)

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

    def find_diff_in_words(self, target_word: str, string: str) -> tuple[str, int]:
        #print("finding the difference, target word is ", target_word)
        partially_constructed = ""
        index = 0
        for letter in target_word:
            partially_constructed += letter
            if string.find(partially_constructed) != -1:
                index += 1
                continue
            #print(letter, "letter not found")
            return (letter, index)
        print("No difference found")
        return ()

    def find_valid_function_token_ids(self, bucket: tuple[str, int], output: str) -> list[int]:
        print("Going into function token ids", bucket)
        valid_tokens: list[str] = []
        further_validation_tokens: list[tuple[str, int]] = []
        predicted_left = self.prediction[(self.prediction.find("<function>") + len("<function>")):]
        func_names = [func.name for func in self.functions]
        if bucket[0] != "<":
            print("found < in the bucket")
            for token in self.token_dict[bucket[0]]:
                length = len(token)
                index_of_temp = output[bucket[1]: bucket[1] + length].find("<")
                if index_of_temp != -1:
                    further_validation_tokens.append((token, index_of_temp))
                elif output.find(token) == 0:
                    valid_tokens.append(token)
            for func in func_names:
                for token, index in further_validation_tokens:
                    if not self.find_diff_in_words(token, output[bucket[1]: index] + func + predicted_left):
                        valid_tokens.append(token)
            return self.convert_token_to_id(valid_tokens)
        cur_func_output = output[(output.find(Key.NAME) + len(Key.NAME)):].replace("Ġ", "")
        print("cur_func_out:", cur_func_output)
        for func in func_names:
            letter = self.find_diff_in_words(func, cur_func_output)
            for token in self.token_dict[letter]:
                if not self.find_diff_in_words(cur_func_output + token, func + predicted_left):
                    valid_tokens.append(token)
        return self.convert_token_to_id(valid_tokens)

    def find_valid_token_ids_in_bucket(self, bucket: tuple[str, int], output: str) -> list[int]:
        valid_tokens : list[int] = []
        print("Bucket still is", bucket)
        if bucket[0] not in self.token_dict.keys():
            raise KeyError(f"Error: No '{bucket[0]}' key in token_dictionary")
        for token in self.token_dict[bucket[0]]:
            if self.find_diff_in_words(token, self.prediction[bucket[1]:]) == "<":
                index = self.prediction[bucket[1]:].find("<")
                if self.prediction[index:].find("<function>") == 0:
                    print("sending to function_token_ids")
                    return self.find_valid_function_token_ids(bucket, output)
            elif self.prediction[bucket[1]:].find(token) == 0:
                valid_tokens.append(token)
        print("valid:")
        print([token for token in valid_tokens])
        return self.convert_token_to_id(valid_tokens)

    def find_general_bucket(self, current_output: str) -> (list[int] | None):
        bucket = self.find_diff_in_words(self.prediction, current_output)
        if not bucket:
            print("Nothing in bucket")
            return None
        if bucket[0] == "<":
            print("found < in bucket")
            if (Key.NAME in current_output
               and self.prediction.find("<function>") == bucket[1]):
                found = False
                for func in self.functions:
                    if self.prediction[bucket[1]:].find(func.name) == 0:
                        found = True
                if found is True:
                    self.prediction.replace("<function>", func.name)
                else:
                    return self.find_valid_function_token_ids(bucket, current_output)
            elif (Key.PARAM in current_output
                 and self.prediction.find("<parameters>") == bucket[1]):
                  """Send to param config"""
            else:
                raise ValueError("Error: Could not find name")
        print("Found bucket", bucket[0])
        return self.find_valid_token_ids_in_bucket(bucket, current_output)

    def correct_logits(self, logits: list[float], cur_output: str) -> (list[float] | None):
        print("\nNew logits:")
        print(self.prediction)
        valid_token_ids = self.find_general_bucket(cur_output.replace(" ", "Ġ"))
        if valid_token_ids is None:
            return None
        print(valid_token_ids)
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
        self.token_dictionary: dict[str, dict[str, int]] = {}
        self.create_token_to_token_id_dict()
        self.const_decode = ConstrainedDecoding(self.token_dictionary,
                                                functions)
        eng_text = EngeneerTextFormat(functions)
        self.engeneered_text = eng_text.create_llm_prompt
        self.encoded_output: list[int] = []

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

    def encode_text_gen(self) -> Generator[list[int], None, None]:
        """
        Encodes the engeneered text into token ids for the llm to process.
        """
        for text in self.prompts:
            llm_text = self.engeneered_text(text)
            self.const_decode.update_prompt(text)
            print(llm_text)
            yield self.llm.encode(llm_text).tolist()[0]

    def token_selection(self) -> float:
        """
        Chooses best token based off llm's probability and constrained decoding
        """
        logits = self.llm.get_logits_from_input_ids(self.encoded)
        if len(logits) == 0:
            raise ValueError("Error: No tokens were found")
        output = self.llm.decode(self.encoded_output)
        print(f"current output '{output}'")
        cor_logits = self.const_decode.correct_logits(logits, output)
        if cor_logits is None:
            return -1
        best_token_id = int(np.argmax(cor_logits))
        return best_token_id

    def prompt_process(self) -> None:
        """
        For each prompt in file it sends the prompt to the necessary functions
        so that it may be encoded, tokenised, logitised and produce the
        desired json output for each prompt to write to the output file.
        """
        encoded_gen = self.encode_text_gen()
        self.encoded = next(encoded_gen)
        i = 0
        while True:
            print("Choosing next token round:", i)
            next_token_id = self.token_selection()
            if next_token_id == -1:
                break
            self.encoded.append(next_token_id)
            print(f"LLM has chosen: '{self.llm.decode(next_token_id)}'")
            self.encoded_output.append(next_token_id)
            i += 1
            if i == 25:
                break
        print("\nLLM response:")
        print(f"'{self.llm.decode(self.encoded_output)}'")
from llm_sdk import Small_LLM_Model
from .parser import FunctonDefinition
from collections.abc import Generator
import numpy as np
import json
import sys


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
        self.target_words = ['{', '"prompt":', '"name":', '"parameters":', '}']
        self.prompt = ""

    def set_invalids_to_infinity(self, logits: list[float]) -> list[float]:
        useful_logits: list[float] = []
        for token_id in self.useful_token_ids:
            useful_logits.append(logits[token_id])
        corrected_logits = [float("-inf")] * len(logits)
        index = 0
        for token_id in self.useful_token_ids:
            corrected_logits[token_id] = useful_logits[index]
            index += 1
        return corrected_logits

    def find_useful_token_ids(self, target_list: list[str]) -> None:
        self.useful_token_ids: list[int] = []
        for target in target_list:
            for token_id in self.token_dict[target].values():
                self.useful_token_ids.append(token_id)
        if len(self.useful_token_ids) == 0:
            raise ValueError("Error: No valid tokens were found.")

    def find_diff_in_words(self, target_word: str, output: str) -> str:
        print("finding the difference, target word is ", target_word)
        for letter in target_word:
            if output.find(letter) != -1:
                continue
            print(letter, "letter not found")
            return letter
        print("No difference found")
        return ""

    def check_value_completed(self, line: str) -> list[str]:
        diff: list[str] = []
        if ":" not in line:
            return [":"]
        for type, value in line.split(":"):
            if type == self.target_words[0]:
                diff.append(self.find_diff_in_words(self.prompt, value))
            elif type == self.target_words[1]:
                for func in self.functions.name:
                    diff.append(self.find_diff_in_words(func, value))
        if "" in diff:
            diff = []
        return diff

    def json_commas(self, line: str) -> bool:
        if "," not in line and ("prompt" in line or "name" in line):
            return False
        return True

    def predict_target_tokens(self, current_output: str, prompt: str) -> int:
        if self.prompt != prompt:
            self.prompt = prompt
        target_bucket: list[str] = []
        lines: list[str] = []
        if "{" in current_output:
            lines = ["{"]
            current_output = current_output.replace("{", "", 1)
        lines = lines + list(current_output.split(","))
        print("Current lines", lines)
        index = 0
        for word in self.target_words:
            print("current word:", word)
            if word == self.target_words[3] and len(lines) == 3:
                print("found function, moving to params")
                _, self.func_name = lines[2].split(":")
            print("Checking if word is not in lines")
            if word not in lines[index]:
                print("Could not find word in line: '", lines[index], "'")
                target_bucket.append(self.find_diff_in_words(word, lines[index]))
                break
            elif word in lines[index] and word != "{":
                difference = self.check_value_completed(lines[index])[0]
                print("Found word and checking difference", difference)
                if difference:
                    target_bucket = difference
                    break
            if self.json_commas == True or word == "{" or word == "}":
                index += 1
            else:
                target_bucket = ","
                break
        print("target bucket is", target_bucket)
        if not target_bucket[0]:
            print("Found everything needed, now exiting")
            return 1
        print("finding the useful token_ids")
        self.find_useful_token_ids(target_bucket)
        return 0

    def correct_logits(self, logits: list[float], cur_output: str,
                       cur_prompt: str ) -> list[float]:
        print("\nNew logits:")
        if self.predict_target_tokens(cur_output, cur_prompt) == 1:
            return []
        return self.set_invalids_to_infinity(logits)


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
        eng_text = EngeneerTextFormat(functions)
        self.engeneered_text = eng_text.create_llm_prompt
        self.const_decode = ConstrainedDecoding(self.token_dictionary,
                                                functions)
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
            self.cur_prompt = text
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
        print("current output", output)
        cor_logits = self.const_decode.correct_logits(logits, output,
                                                      self.cur_prompt)
        if not cor_logits:
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
            next_token_id = self.token_selection()
            if next_token_id == -1:
                break
            self.encoded.append(next_token_id)
            self.encoded_output.append(next_token_id)
            i += 1
            if i == 30:
                break
        print("\nLLM response:")

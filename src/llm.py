from llm_sdk import Small_LLM_Model
from .parser import FunctonDefinition
from collections.abc import Generator
import json


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
    def __init__(self) -> None:
        return

    def search_and_invalidate(self, logits: list[float], target: int) -> None:
        return

    def prompt_formation(self) -> None:
        return

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
        self.create_token_to_token_id_dict()
        eng_text = EngeneerTextFormat(functions)
        self.engeneered_text = eng_text.create_llm_prompt
        self.encoded_ouput: list[int] = []

    def create_token_to_token_id_dict(self) -> None:
        """
        Creates a dictionary to allow efficient look up of token id to token
        """
        with open(self.llm.get_path_to_vocab_file()) as f:
            token_to_token_id = json.load(f)
        self.token_id_to_token = {value: key
                                  for key, value in token_to_token_id.items()}

    def encode_text_gen(self) -> Generator[list[int], None, None]:
        """
        Encodes the engeneered text into token ids for the llm to process.
        """
        for text in self.prompts:
            llm_text = self.engeneered_text(text)
            print(llm_text)
            yield self.llm.encode(llm_text).tolist()[0]

    def token_selection(self) -> float:
        """
        Chooses best token based off llm's probability and constrained decoding
        """
        logits = self.llm.get_logits_from_input_ids(self.encoded)
        if len(logits) == 0:
            raise ValueError("Error: No tokens were found")
        logits_array = zip(range(len(logits)), logits)
        print("sorting")
        ordered_logits = sorted(logits_array, key=lambda pair: pair[1],
                                reverse=True)
        print("done sorting")
        best_token_id = ordered_logits[0][0]
        return best_token_id

    def token_id_to_text(self, token_ids = list[int]) -> str:
        """
        Converts token id to token
        """
        text = ""
        for token_id in token_ids:
            text += self.token_id_to_token[token_id]
        return text

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
            self.encoded.append(next_token_id)
            self.encoded_ouput.append(next_token_id)
            if self.token_id_to_token[next_token_id] == "}":
                break
            i += 1
            if i == 30:
                break
        print("\nLLM response:")
        print(self.llm.decode(self.encoded_ouput))

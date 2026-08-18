import json
import argparse
from .parser import load_definition, load_prompts
from .llm import LLMProcessing

def main() -> int:
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--functions_definition",
                             default="data/input/functions_definition.json")
    main_parser.add_argument("--input",
                             default="data/input/function_calling_tests.json")
    main_parser.add_argument("--output", default="data/output/"
                             "function_calls.json")
    args = main_parser.parse_args()
    try:
        function_defs = load_definition(args.functions_definition)
        prompts = load_prompts(args.input)
        print(prompts[0])
        llm = LLMProcessing(prompts, function_defs)
        llm.prompt_process()
    except (ValueError, TypeError) as message:
        print(message)
    return 0

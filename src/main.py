import json
import argparse
from .parser import load_definition, load_prompts
from .llm import LLMProcessing

def main() -> int:
    """
    Main function that calls each neccsary function to run the program.
    Only accepts 3 other arguments if required otherwise default arguments
    are used.
    """
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--functions_definition",
                             default="data/input/functions_definition.json")
    main_parser.add_argument("--input",
                             default="data/input/function_calling_tests.json")
    main_parser.add_argument("--output", default="data/output/"
                             "function_calls.json")
    args = main_parser.parse_args()
    try:
        print("Loading prompts")
        function_defs = load_definition(args.functions_definition)
        prompts = load_prompts(args.input)
        print(prompts[0])
        print("Creating the llm processor")
        llm = LLMProcessing(prompts, function_defs)
        print("Finshed loading the llm processor")
        print("Starting all prompt processor")
        llm.all_prompt_process()
    except (ValueError, TypeError, KeyError) as message:
        print(message)
    return 0

if __name__ == "__main__":
    main()
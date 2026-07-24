import json
import argparse
from .definition_parser import load_file

def main() -> int:
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument("--functions_definition",
                             default="data/input/functions_definition.json")
    main_parser.add_argument("--input",
                             default="data/input/function_calling_tests.json")
    main_parser.add_argument("--output", default="data/output/"
                             "function_calls.json")
    args = main_parser.parse_args()
    load_file(args.functions_definition)
    return 0

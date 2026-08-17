from llm_sdk.llm_sdk import Small_LLM_Model as llm
from typing import Any
class logits():
    def __init__(self):
        self.path = llm.get_path_to_vocab_file()

    def encoding(prompt: str) -> Any:
        encoded = llm.encode(prompt)

    def convert_token_to_ids(Any) -> :
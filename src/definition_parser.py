import json
from pydantic import BaseModel, Field, ConfigDict
from pydantic import model_validator  # type: ignore[attr-defined]
from typing import Literal

class ParameterDefinition(BaseModel):
    type: Literal["string", "number"]
    model_config = ConfigDict(extra="forbid", strict=True)

class FunctonDefinition(BaseModel):
    name: str = Field(min_length=3)
    description: str = Field(min_length=5)
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode='after')
    def validate_name(self) -> FunctonDefinition:
        if " " in self.name or "-" in self.name or "." in self.name:
            raise ValueError("Error: Function name cannot have any of the"
                             "following characters: ' ', '.' or '-'.")
        elif (self.name).isnumeric():
            raise ValueError("Error: Function name cannot start with a number.")
        return self
        
def load_file(filename: str) -> int:
    with open(filename) as f:
        all_defs = json.load(f)
    function_defs: list[FunctonDefinition] = []
    for functions in all_defs:
        function_defs.append(FunctonDefinition(**functions))
    for i in function_defs:
        print("name:", i.name, "\ndescription:", i.description, "\nparameters",
              i.parameters, "\nreturns", i.returns.type)



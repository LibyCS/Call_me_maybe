import json
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from pydantic import model_validator  # type: ignore[attr-defined]
from pydantic_core import PydanticCustomError
from typing import Literal

class ParameterDefinition(BaseModel):
    """
    Pydantic model that makes sure the paramater type can only be
    string or a number, aswell as ensuring its formatted correctly.

    type: Literal['string', 'number']
    """
    type: Literal["string", "number"]

    @model_validator(mode="before")
    @classmethod
    def check_fields(cls, data: any) -> any:
        """
        Checks that data has been provided with the correct field. No
        additional fields are permitted.
        """
        if not isinstance(data, dict):
            raise PydanticCustomError("misiing_dict", "Must be a dictionary")
        elif len(data) == 0:
            raise PydanticCustomError("no_field_found",
                                      "Must have the 'type' field")
        elif len(data) > 1:
            raise PydanticCustomError("too_many_fields",
                                      "Only 1 field is allowed "
                                      "for each parameter")
        elif "type" not in data:
            raise PydanticCustomError("wrong_field",
                                      "Only 'type' field is allowed")
        return data
        
class FunctonDefinition(BaseModel):
    """
    Main pydantic model for each function and their definition, parameters
    are validated and stored for future use.
    
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition
    """
    name: str = Field(min_length=3)
    description: str = Field(min_length=5)
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data: any) -> any:
        """
        Validates each field to ensure all necessary fields are present
        """
        for fields in ["name", "description", "parameters", "returns"]:
            if fields not in data:
                raise PydanticCustomError("missing_field",
                                          "A function must have the fields "
                                          "'name', 'description', 'parameters'"
                                          " and 'returns'")
        if len(data) > 4:
            raise PydanticCustomError("too_many_fields",
                                      "Too many fields were given")
        return data
    @model_validator(mode="after")
    def validate_name(self) -> "FunctonDefinition":
        """
        Validates name of the provided function so that it is not invalid
        """
        if " " in self.name or "-" in self.name or "." in self.name:
            raise PydanticCustomError("invalid_name",
                                      "Function name cannot have any of the"
                                      "following characters: ' ', '.' or '-'.")
        elif (self.name[0]).isnumeric():
            raise PydanticCustomError("invalid_name",
                                      "Function name cannot start"
                                      " with a number.")
        return self

def load_definition(filename: str) -> list[FunctonDefinition]:
    """
    Opens the function definition JSON file and passes each of them
    through the pydantic FunctionDEfiniton Model to validate and
    store them in a list named function_defs.
    Returns the list of all pydantic functions.
    """
    with open(filename) as f:
        all_defs = json.load(f)
    function_defs: list[FunctonDefinition] = []
    for functions in all_defs:
        try:
            new_function = FunctonDefinition(**functions)
            for function in function_defs:
                if new_function.name == function.name:
                    raise ValueError("Error: 2 functions have the same name")
            function_defs.append(new_function)
        except ValidationError as message:
            for error in message.errors():
                print(f"Error in {filename}:")
                raise ValueError(f"Error: {error["msg"]}")
    for i in function_defs:
        print("name:", i.name, "\ndescription:", i.description, "\nparameters",
              i.parameters, "\nreturns", i.returns.type)
    return function_defs

def load_prompts(filename: str) -> list[str]:
    """
    Opens the prompts JSON file and stores all prompts as strings in a list.
    Returns the list."""
    with open(filename) as f:
        data = json.load(f)
    prompts: list[str] = []
    for prompt in data:
        prompts = prompts + list(prompt.values())
    print(prompts)
    return prompts
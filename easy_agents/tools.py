from pydantic import BaseModel, Field
from typing import Callable


class Tool(BaseModel):
    name: str = Field(..., description="The name of the tool.")
    description: str = Field(
        ..., description="A brief description of what the tool does."
    )
    rules: str = Field(
        ..., description="Rules for using the tool used to guide the agent."
    )
    function: Callable = Field(
        ..., description="The function to be called when the tool is used."
    )

    class Config:
        arbitrary_types_allowed = True

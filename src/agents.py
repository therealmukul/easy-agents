import re

import openai
import httpx

from src.llm.openai import OpenAI

from pydantic import BaseModel, Field
from typing import Callable


class Tool(BaseModel):
    name: str = Field(..., description="The name of the tool")
    description: str = Field(
        ..., description="A brief description of what the tool does"
    )
    function: Callable = Field(
        ..., description="The function to be called when the tool is used"
    )

    class Config:
        arbitrary_types_allowed = True


class Agent:
    def __init__(self, llm: OpenAI, max_turns=5, system_prompt: str = None):
        self.llm = llm
        self.max_turns = max_turns
        self.messages = []
        self.tools = {}
        self.actions_re = re.compile("^Action: (\w+): (.*)")

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def execute(self, message: str):
        self.messages.append({"role": "user", "content": message})
        results = self.llm.send_message(self.messages)
        response_text = results[0]
        self.messages.append({"role": "assistant", "content": response_text})

        return response_text

    def run(self, message: str):
        i = 0
        next_message = message

        while i < self.max_turns:
            i += 1
            agent_response = self.execute(next_message)

            print(agent_response)

            actions = [
                self.actions_re.match(a)
                for a in agent_response.split("\n")
                if self.actions_re.match(a)
            ]

            if actions:
                action, actions_input = actions[0].groups()
                
                if action in self.tools:
                    print(f'-' * 10)
                    print(f'Tool call:')
                    print(f'\tName: {action}')
                    print(f'\tInput: {actions_input}')
                    print(f'-' * 10)

                    observation = self.tools[action].function(actions_input)
                    print(f"Observation: {observation}")
                    next_message = f"Observation: {observation}"
            else:
                return agent_response


class BaseAgent:

    def __init__(
        self, client: openai.OpenAI, model="gpt-4o-mini", system_prompt: str = ""
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.messages = []

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def __call__(self, message: str):
        self.messages.append({"role": "user", "content": message})
        results = self.execute()
        self.messages.append({"role": "assistant", "content": results})

        return results

    def execute(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        return response.choices[0].message.content

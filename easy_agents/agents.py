import re

from easy_agents.llm.openai import OpenAI
from easy_agents.tools import Tool

from pydantic import BaseModel, Field
from typing import Callable


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
                    print(f"-" * 10)
                    print(f"Tool call:")
                    print(f"\tName: {action}")
                    print(f"\tInput: {actions_input}")
                    observation = self.tools[action].function(actions_input)
                    print(f"\tOutput: {observation}")
                    print(f"-" * 10)

                    print(f"Observation: {observation}")
                    next_message = f"Observation: {observation}"
            else:
                return agent_response

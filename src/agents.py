import re

import openai
import httpx


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

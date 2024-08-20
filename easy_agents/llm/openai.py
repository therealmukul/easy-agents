from typing import Dict, List, Union
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai import Stream

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt


class OpenAI:
    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI()

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def send_message(
        self, messages: List[Dict[str, str]]
    ) -> (str, (ChatCompletion | Stream[ChatCompletionChunk])):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )

        res = response.choices[0].message.content

        return res, response

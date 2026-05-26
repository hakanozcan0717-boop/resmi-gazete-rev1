# -*- coding: utf-8 -*-

import os
from openai import OpenAI


class LLMClient:
    def __init__(self, model: str = None):
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY bulunamadı.")

        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def generate_answer(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Sen Resmî Gazete belgelerine göre Türkçe cevap veren bir analiz asistanısın."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content
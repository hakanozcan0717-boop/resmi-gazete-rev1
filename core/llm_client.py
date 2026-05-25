# -*- coding: utf-8 -*-
"""
LLM Client Modülü

Bu dosya OpenAI API ile iletişim kurar.
RAG sisteminin bulduğu kaynak metinleri modele gönderir
ve kaynaklara dayalı cevap üretir.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


# .env dosyasındaki OPENAI_API_KEY değerini yükler.
load_dotenv()


class LLMClient:
    """
    OpenAI LLM istemcisi.

    Bu sınıfın görevi:
    - API anahtarını okumak
    - OpenAI istemcisini oluşturmak
    - Prompt gönderip cevap almak
    """

    def __init__(self, model: Optional[str] = None):
        """
        model:
            Kullanılacak OpenAI modeli.
            .env içinde OPENAI_MODEL varsa onu kullanır.
            Yoksa varsayılan olarak gpt-5.2 kullanır.
        """
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.2")

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY bulunamadı. Proje klasöründe .env dosyası oluşturup "
                "içine OPENAI_API_KEY=... yazmalısın."
            )

        self.client = OpenAI(api_key=api_key)

    def generate_answer(
        self,
        prompt: str,
        instructions: Optional[str] = None,
    ) -> str:
        """
        LLM'e prompt gönderir ve cevabı döndürür.

        prompt:
            Modele gönderilecek ana içerik.

        instructions:
            Modelin nasıl davranacağını belirleyen genel talimat.
        """
        if instructions is None:
            instructions = (
                "Sen bir Resmî Gazete analiz asistanısın. "
                "Sadece verilen kaynak metinlere dayanarak cevap ver. "
                "Kaynaklarda olmayan bilgiyi uydurma. "
                "Cevabını Türkçe, açık ve düzenli yaz. "
                "Gerekirse maddeler halinde açıkla."
            )

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )

        return response.output_text

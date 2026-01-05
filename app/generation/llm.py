import json,os

from google import genai
from google.genai import types
from app.core.cache import RedisClient
from app.retrieval.utils import hash_query
client = genai.Client(
        api_key=os.getenv("GENAI_API_KEY"),
    )
class GeminiLLM:
    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        max_context_chars: int = 8000,
    ):
         # set at call time
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.redis = RedisClient.get_client()

    def _cache_key(self, query: str) -> str:
        qh = hash_query(query)
        return f"final:{self.model_name}:{qh}"

    def _build_context(self, chunks):
        context = ""
        for c in chunks:
            if len(context) + len(c["text"]) > self.max_context_chars:
                break
            context += c["text"] + "\n\n"
        return context.strip()

    def generate(self, query: str, chunks):
        cache_key = self._cache_key(query)

        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        context = self._build_context(chunks)

        prompt = f"""
You are a helpful AI assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
        response = client.models.generate_content(
        model=self.model_name,
        contents=contents,

    )
        
        answer = response.candidates[0].content.parts[0].text.strip()

        self.redis.setex(
            cache_key,
            600,  # 10 min TTL
            json.dumps(answer),
        )

        return answer


# To run this code you need to install the following dependencies:
# pip install google-genai



import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

MODEL = "google/gemma-4-31b-it:free"

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)

docs = [
    "The X product has 2 years of warranty and a 30-day money back guarantee.",
    "The Y product has a 3 year warranty and a 20-day money back guarantee.",
    "The Z product has a 5 year warranty and a 10-day money back guarantee.",
    "To cancel your subscription, access Configurations > Account > Cancel Subscription.",
    "The plan is free for the first year, after which it costs $19.99 per month.",
    "You can cancel your subscription at any time by logging into your account and going to Configurations > Account > Cancel Subscription.",
]


def embeddings(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([e.embedding for e in resp.data])


def get_indice():
    return embeddings(docs)


def find(question: str, k: int = 2) -> list[str]:
    index = get_indice()
    question_vector = embeddings([question])[0]

    # cosine similarity
    norms = np.linalg.norm(index, axis=1) * np.linalg.norm(question_vector)
    similarities = (index @ question_vector) / norms

    top_k = np.argsort(similarities)[::-1][:k]
    return [docs[i] for i in top_k]


def rag(question: str) -> str | None:
    context = find(question, k=2)
    context_str = "\n".join(f"- {doc}" for doc in context)

    prompt = f"""
    You are a helpful assistant that answers questions based on a given context.
    if you don't know the answer, just say that you don't know. Don't try to make up an answer.
    
    Context:
    {context_str}
    
    Question: {question}
    """

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return f"Question: {question}\nAnswer: {resp.choices[0].message.content}\n\n"

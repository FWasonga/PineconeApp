# # query.py
# import os
# from groq import Groq

# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")  # ← paste your key here

# client = Groq(api_key=GROQ_API_KEY)

# def generate_answer(question: str, chunks: list[str]) -> str:
#     context = "\n\n".join(chunks)
    
#     response = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant. Answer questions based only on the provided context. If the answer is not in the context, say 'I don't know'."
#             },
#             {
#                 "role": "user",
#                 "content": f"Context:\n{context}\n\nQuestion: {question}"
#             }
#         ]
#     )
    
#     return response.choices[0].message.content

import os
from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions based only on the provided context. If the answer is not in the context, say 'I don't know'."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    
    return response.choices[0].message.content
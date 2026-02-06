# agents/external.py

# async def external_answer(question: str, model_client) -> str:
#     """
#     Uses the LLM directly for general knowledge questions
#     NOT answerable from documents.
#     """

#     prompt = f"""
# You are answering using general world knowledge.

# Question:
# {question}

# Rules:
# - Be factual
# - If unsure, say so clearly
# - Do not reference internal documents
# """

#     # response = await model_client.create(
    #     messages=[{"role": "user", "content": prompt}]
    # )

    # return response.choices[0].message.content.strip()

# # agents/external.py

# from autogen_agentchat.agents import AssistantAgent

# EXTERNAL_SYSTEM_PROMPT = """
# You are an External Knowledge Agent.

# Rules:
# - Use general world knowledge.
# - Be factual and concise.
# - If unsure, say so.
# - Do NOT mention internal documents.
# """

# def create_external_agent(model_client):
#     return AssistantAgent(
#         name="ExternalAgent",
#         model_client=model_client,
#         system_message=EXTERNAL_SYSTEM_PROMPT,
#     )

from autogen_agentchat.agents import AssistantAgent

EXTERNAL_SEARCH_PROMPT = """
You are an External Knowledge Agent.

Rules:
- You may use general world knowledge.
- You may answer questions not covered by internal documents.
- Be factual, concise, and clear.
- If you are unsure, say so.
- Do NOT mention internal documents or RAG.
"""

def create_external_agent(model_client):
    """
    Factory function to create an External Knowledge Agent.
    """
    return AssistantAgent(
        name="ExternalAgent",
        model_client=model_client,
        system_message=EXTERNAL_SEARCH_PROMPT,
    )

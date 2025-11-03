from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

import os

MODEL = os.environ["MODEL"]
TEMP = int(os.environ["MODEL_TEMP"])
MAX_TOKENS = int(os.environ["MODEL_MAX_TOKENS"])


def generate_response_from_conversation(conversation: str) -> str:
    """Use Groq to run inference and return the response based on the conversation
    built by the coordinator.

    Args:
        conversation (str): The complete conversation history + retrieved documents with prompt
        for answering.

    Returns:
        str: The answer returned by the LLM model.
    """
    llm = ChatGroq(
        model=MODEL,
        temperature=TEMP,
        max_tokens=MAX_TOKENS,
    )
    messages = [HumanMessage(content=conversation)]
    response = llm.invoke(messages)

    if isinstance(response, list):
        if len(response) > 0:
            content = (
                response[0].get("content", "")
                if isinstance(response[0], dict)
                else str(response[0])
            )
        else:
            content = ""
    else:
        content = getattr(response, "content", str(response))

    return content.strip()
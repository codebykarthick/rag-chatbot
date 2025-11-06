from typing import Any

from rag.generator import generate_response_from_conversation
from rag.retriever import retrieve_from_vector_store


def escape_markdown(text: str) -> str:
    return text.replace("$", "\\$")


def retrieve_and_generate(messages: Any, prompt: str) -> str:
    """Iterate through our chat history to build up the context to retrieve relevant docs and 
    generate a summary with LLM.

    Args:
        messages (Any): The list of messages in the chat history.
        prompt (str): The current prompt that the user entered.

    Returns:
        str: The response generated through the coordinated retrieval from vector stores 
        and generation by the LLM.
    """
    # Send the prompt as a query to the retriever
    documents = retrieve_from_vector_store(query=prompt)

    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. Generate answers based on context from past conversation and retrieved documents only."
    for dict_message in messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    """TODO: Think about using short memory instead of building the entire convo history
    cause this is not efficient, although more accurate.
    """

    """TODO: Think about caching practices somewhere to make this efficient. Review bottlenecks.
    """
    
    retrieved_text = "\n\n".join([doc.page_content for doc in documents])
    string_dialogue += "\nRelevant context from retrieved documents:\n" + retrieved_text + "\n\n"
    string_dialogue += f"User: {prompt}\n\nAssistant:"

    print(f"Prompt to generator: \n{string_dialogue}")

    output = generate_response_from_conversation(string_dialogue)

    print(f"Response from generator: \n{output}")
    output = escape_markdown(output)

    return output

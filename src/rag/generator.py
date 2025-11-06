from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import os
from typing import List

load_dotenv()

MODEL = os.environ["MODEL"]
TEMP = float(os.environ["MODEL_TEMP"])
MAX_TOKENS = int(os.environ["MODEL_MAX_TOKENS"])


def generate_company_from_query(query: str) -> List[str]:
    """Function to idenitfy what companies need to be focussed for a given query

    Args:
        query (str): The query provided by the user

    Returns:
        List[str]: The list of companies that we need to focus on.
    """
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,  # make it deterministic for classification
        max_completion_tokens=MAX_TOKENS,
        model_kwargs={"response_format": {"type": "text"}}
    )

    system_prompt = (
        "You are a classification model. You determine which companies are referenced in a question. "
        "The only valid outputs are exactly one of the following: "
        "Tesla, BMW, Ford, Tesla and BMW, Tesla and Ford, BMW and Ford, or All. "
        "Do not explain, only return the label. "
        "Example outputs: 'Tesla', 'BMW and Ford', 'All'."
    )

    messages = [
        HumanMessage(content=f"{system_prompt}\n\nQuestion: {query}")
    ]
    response = llm.invoke(messages)

    content = getattr(response, "content", str(response)).strip()
    print(f"Company classification raw output: {content}")

    # Normalize and convert to list
    companies = []
    if "tesla" in content.lower():
        companies.append("tesla")
    if "bmw" in content.lower():
        companies.append("bmw")
    if "ford" in content.lower():
        companies.append("ford")
    if not companies or "all" in content.lower():
        companies = ["tesla", "bmw", "ford"]

    print(f"Detected companies: {companies}")
    return companies


def generate_response_from_conversation(conversation: str) -> str:
    """Use OpenAI API to run inference and return the response based on the conversation
    built by the coordinator.

    Args:
        conversation (str): The complete conversation history + retrieved documents with prompt
        for answering.

    Returns:
        str: The answer returned by the LLM model.
    """
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMP,
        max_completion_tokens=MAX_TOKENS,
        model_kwargs={"response_format": {"type": "text"}}
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


def generate_search_terms_from_query(query: str, companies: List[str]) -> List[str]:
    """For better query handling use the LLM to determine what information is
    needed to be retrieved for the best chance of answering the query.

    Args:
        query (str): The query posted by the user.
        companies (List[str]): The companies identified in the previous

    Returns:
        List[str]: The queries to be used for fetching relevant content.
    """
    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMP,
        max_completion_tokens=MAX_TOKENS,
        model_kwargs={"response_format": {"type": "text"}},
    )

    joined_companies = ", ".join([c.capitalize() for c in companies])
    system_prompt = (
        f"You are a financial analyst assistant. The user is asking a question related to {joined_companies}. "
        "Your task is to list 3-6 concise search terms that would help retrieve relevant data "
        "from annual reports (like revenue, profit, EBIT, or growth). Avoid explanations — just list terms, separated by commas. "
        "For example, output like: 'Tesla revenue 2023, Tesla profit 2023, Tesla EBIT margin'."
    )

    messages = [HumanMessage(content=f"{system_prompt}\n\nQuestion: {query}")]
    response = llm.invoke(messages)

    content = getattr(response, 'content', str(response)).strip()
    print(f"Search term generation raw output: {content}")

    terms = [term.strip() for term in content.replace(';', ',').split(',') if term.strip()]
    print(f"Parsed search terms: {terms}")
    return terms

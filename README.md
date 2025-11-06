# RAG-Powered ChatBot

## Introduction
RAG ChatBot implemented with a StreamLit dashboard in a docker container.

## Setup
1. It is recommended to create a fresh virtual environment. The dependencies are installed using the below command.

```bash
# Install dependencies
pip3 install -r requirements.txt
```

2. Input your config in `.env` file which will be used by the application. Use `.env.sample` for needed args.

3. The vectors need to be precomputed (to prevent repetition), done using the command below.

```bash
python3 scripts/preprocess_docs.py
```

which creates the vectors and index at `./Data/` alongside the documents.

4. Run the dashboard locally with the below.

```bash
# Run the dashboard
cd src && streamlit run app.py
```

Project also supports a docker container by building and running with the following commands. This does not need the above commands as it automatically runs all the commands needed (except for .env configuration).

```bash
# Build the docker image
docker build -t rag-chatbot .

# Run the image
docker run -p 8501:8501 rag-chatbot
```

## Usage

A streamlit dashboard page is opened at 8501 port on the deployment target, exposing a simple chat interface for usage. Logs are written to stdout of the application.


## Architecture Overview
The chatbot follows a three-stage RAG (Retrieval-Augmented Generation) workflow:
1. **Classification:** Detects which company or combination the query refers to.
2. **Retrieval Planning:** Expands the user query into structured search terms.
3. **Retrieval + Generation:** Retrieves relevant financial report sections and synthesizes the final answer using GPT‑5 (via OpenAI API).

## Example Queries
| Query | Model Response (shortened) |
|-------|-----------------------------|
| "What was Tesla’s total revenue in 2023?" | Tesla generated ~$96.77B in 2023 |
| "Compare Tesla and Ford’s profitability in 2022." | Tesla reported higher net income and operating margins than Ford in 2022. |

## Error Handling
- The system handles exceptions gracefully. If an error occurs during inference or retrieval, a fallback message ("Error occurred, try again later.") is displayed instead of crashing.

## Future Work
Explained in detail in the below section.


## Analysis
### 1. Precomputation of Vectorisation:
---
1.1. The documents were parsed in a naive manner using sequential chunking. This approach works adequately for Tesla and Ford annual reports, which have a relatively simple, linear structure. However, it performs poorly on BMW reports due to their complex layout and formatting, suggesting that additional pre-processing (such as layout-aware parsing or table handling) would significantly improve retrieval quality.

1.2. Due to number of documents, the vectors were pre-computed and loaded in memory, to prevent access latency. For a real world application, a separate vector DB server can be hosted instead.

### 2. Retrieval:
---
2.1. Right now, due to time constraints, retrieval is only based off of FAISS vector cosine similarity. This can be extended to a hybrid retrieval stance - Semantic + keyword/BM25 to better catch direct matchs in facts.

2.2. Better chunking with semantics can be done - split by headings / paragraphs and not just fixed characters to keep relevant concepts together. Also can do hierarchical chunking.

2.3. Metadata tagging during ingestion can be performed for easier filtering before actual vector search.

### 3. Generation:
---
3.1. Right now generation is used in three ways - to determine which company the natural query is talking about (actual company may not be in the query explicitly), which is easily identified by the LLM with query + conversation history.

3.2. Using LLMs to generate the actual terms to search also makes it much more flexible to accept natural language queries.

3.3. The entire conversation history is fed as context - while it is accurate, costs / context window limitation of models mean the conversation has to be either truncated and/or summarised into a short memory form to prevent inaccuracies.

3.4. Session is not persisted and can be cached for faster responses.
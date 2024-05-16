# non-proprietry-llm

The repo leverages langchain, google gemini ai and stream lit

PDFs pre processes with pypdf.
Embeddings are generated with gemini model and indexed them to faiss.

Using Langchain runnable sequences RAG chain is implemented and fetched relevant docs from Faiss and fed to gemini together with user query and relevant context.

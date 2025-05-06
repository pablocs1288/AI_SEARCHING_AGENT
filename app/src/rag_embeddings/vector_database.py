import os

from langchain.vectorstores import Chroma # ChromaDB Langchain wrapper supports hybrid search

from src.rag_embeddings.local_embeddings import Embeddings

class VectorDB:

    def __init__(self):
        embeddings = Embeddings(os.environ['LOCAL_EMBEDDING_MODEL'])
        self.vectordb = Chroma(embedding_function=embeddings, persist_directory=os.environ['CHROMA_DB_DIR'])

    def add_documents(self, documents, metadata=None):
        """ Add new text chunks into the vector database """
        #vectordb.add_texts(text_chunks, metadatas=[metadata]*len(text_chunks) if metadata else None)
        self.vectordb.add_documents(documents=documents)

    def retrieve_from_vector_db(self, query: str, company_name: str):
        """ Retrieve relevant documents based on query """
        docs_with_scores = self.vectordb.similarity_search_with_score(
            query, 
            k=20, # see it later
            filter={"company_name": company_name}
            ) # filter param allows hybrid search= vector + metadata querying
        return docs_with_scores
import os

from src.tools.tool import Tool
from src.rag_embeddings.vector_database import VectorDB


class RetrieverTool(Tool):



    def __init__(self, vector_db: VectorDB):
        super().__init__()
        self.vector_db = vector_db


    def invoke_tool(self, company_name: str):

        similarity_treshold = int(os.environ['SIMILARITY_THRESHOLD'])

        prompt = f"Instruct: Given a query represented as a company name, retrieve the names of their board members.\nCompany name: {company_name}." # no está retornando nada
        docs_with_scores = self.vector_db.retrieve_from_vector_db(prompt, company_name)
        
        if docs_with_scores and docs_with_scores[0][1] < similarity_treshold:
            return  "\n".join(doc.page_content for doc, score in docs_with_scores)
       
        return None


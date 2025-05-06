
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.tools.tool import Tool
from src.rag_embeddings.vector_database import VectorDB


class PersistenceTool(Tool):


    def __init__(
            self,
            vector_db: VectorDB,
            chunk_size: int = 400, 
            chunk_overlap: int = 0
            ):
        super().__init__()

        self.vector_db = vector_db
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def invoke_tool(self, text: str, company_name: str):
        
        documents = self._chunk_text(
            text, 
            metadata = {"company_name": company_name}
            )
        
        self.vector_db.add_documents(documents=documents)
        #vectordb.save_local(vector_store_path)


    def _chunk_text(self, text, metadata:dict = None):
        
        """ Split text into chunks for embedding """

        doc = Document(page_content=text, metadata=metadata)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
            ) 
        
        return splitter.split_documents([doc])

from typing import List, Dict, Any
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from config import settings
import os

class RAGSystem:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        """Create the QA chain with a custom prompt"""
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create the retriever
        retriever = self.vector_store.as_retriever()
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT, "document_variable_name": "context"},
            return_source_documents=True
        )
        
        return qa_chain

    def add_documents(self, file_paths: List[str]) -> None:
        """Process and add documents to the vector store"""
        documents = self.document_processor.process_documents(file_paths)
        if documents:
            self.vector_store.add_documents(documents)
            print(f"Successfully processed and added {len(documents)} document chunks.")
        else:
            print("No documents were processed.")

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not question.strip():
            return {"answer": "Please provide a valid question.", "sources": []}
        
        try:
            # Use the invoke method instead of __call__
            result = self.qa_chain.invoke({"query": question})
            
            # Extract sources from the retrieved documents
            sources = list(set([
                doc.metadata.get('source', 'Unknown source') 
                for doc in result.get('source_documents', [])
                if hasattr(doc, 'metadata')
            ]))
            
            return {
                "answer": result.get("result", "No answer found."),
                "sources": sources
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": []
            }
    
    def clear_knowledge_base(self) -> None:
        """Clear all documents from the vector store"""
        self.vector_store.delete_collection()
        print("Knowledge base has been cleared.")

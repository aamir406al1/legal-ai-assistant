from typing import List, Dict, Any, Optional
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage

class RAGEngine:
    def __init__(self, vector_store, model_name="gpt-3.5-turbo"):
        self.vector_store = vector_store
        self.embeddings_model = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    def query(self, query_text: str, doc_filter: Optional[List[str]] = None, top_k: int = 5) -> Dict[str, Any]:
        """Process a query using RAG"""

        print("✅ Entered query()")
        print(f"📨 Query Text: {query_text}")
        # Create embedding for the query
        query_embedding = self.embeddings_model.embed_query(query_text)
        
        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            doc_filter=doc_filter
        )
        
        # Format context for the LLM
        formatted_context = self._format_context(relevant_chunks)
        
        # Generate response with the LLM
        try:
            response = self._generate_response(query_text, formatted_context)
        except Exception as e:
            print("❌ Error in _generate_response:", e)
            response = "An error occurred while generating a response."

        print(f"Query: {query_text}")
        print(f"Chunks retrieved: {len(relevant_chunks)}")
        print(f"Formatted context:\n{formatted_context}")
        print(f"LLM Response: {response}")

        return {
            "query": query_text,
            "response": response or "No relevant answer could be generated.",
            "sources": [
                {
                    "title": chunk["chunk"].metadata.get("title", "Unknown"),
                    "doc_id": chunk["chunk"].metadata.get("doc_id", "Unknown"),
                    "page": chunk["chunk"].metadata.get("page", "Unknown"),
                    "text": chunk["chunk"].page_content[:200] + "..." if len(chunk["chunk"].page_content) > 200 else chunk["chunk"].page_content
                }
                for chunk in relevant_chunks
            ]
        }
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks as context for the LLM"""
        context = "RELEVANT INFORMATION:\n\n"
        
        for i, item in enumerate(chunks):
            chunk = item["chunk"]
            context += f"[DOCUMENT {i+1}] {chunk.metadata.get('title', 'Unknown Document')}\n"
            context += f"Page: {chunk.metadata.get('page', 'Unknown')}\n"
            context += f"Text: {chunk.page_content}\n\n"
        
        return context
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate a response using the LLM with the provided context"""
        system_prompt = (
            "You are a legal assistant for a small business. Your task is to provide accurate "
            "information based on the retrieved legal documents. When answering questions:\n"
            "1. Only use information from the provided documents.\n"
            "2. If the documents don't contain relevant information, say so honestly.\n"
            "3. Include citations to specific documents when appropriate.\n"
            "4. Use clear language appropriate for small business owners, not legal jargon.\n"
            "5. Avoid providing definitive legal advice; rather, focus on information.\n"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context}\n\nQUESTION: {query}\n\nAnswer the question using only the information provided above.")
        ]

        print("✅ _generate_response called")
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else response

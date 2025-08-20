import os
import re
import html
import json
import time
import requests
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument
from markdownify import markdownify
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile

from llama_stack_client import LlamaStackClient
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any, Dict
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from starlette.requests import Request
from ray import serve

# Prompt Templates (defined globally)
summary_template = PromptTemplate(
    input_variables=["document"],
    template="""Create a concise summary of this document in 5-10 sentences:

{document}

SUMMARY:"""
)

facts_template = PromptTemplate(
    input_variables=["document"],
    template="""Extract the most important facts from this document. List them as bullet points:

{document}

KEY FACTS:
-"""
)

qa_template = PromptTemplate(
    input_variables=["document", "question"],
    template="""Based on the following document, answer the question. If the answer isn't in the document, say so.

DOCUMENT:
{document}

QUESTION: {question}

ANSWER:"""
)

class LlamaStackLLM(LLM):
    """Simple LangChain wrapper for Llama Stack"""

    # Pydantic model fields
    client: Any = None
    model_id: str = "llama3.2:3b"

    def __init__(self, client, model_id: str = "llama3.2:3b"):
        # Initialize with field values
        super().__init__(client=client, model_id=model_id)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Make inference call to Llama Stack"""
        response = self.client.inference.chat_completion(
            model_id=self.model_id,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.completion_message.content

    @property
    def _llm_type(self) -> str:
        return "llama_stack"


def load_document(source: str) -> str:
    is_url = source.startswith(('http://', 'https://'))
    is_pdf = source.lower().endswith('.pdf')
    if is_pdf:
        return load_pdf(source, is_url=is_url)
    elif is_url:
        return load_from_url(source)
    else:
        raise ValueError(f"Unsupported format. Use URLs or PDF files.")


def load_pdf(source: str, is_url: bool = False) -> str:
    if is_url:
        response = requests.get(source)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            file_path = temp_file.name
    else:
        file_path = source
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return "\\n\\n".join([doc.page_content for doc in docs])
    finally:
        if is_url:
            os.remove(file_path)


def load_from_url(url: str) -> str:
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; DocumentLoader/1.0)'}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    doc = ReadabilityDocument(response.text)
    html_main = doc.summary(html_partial=True)
    soup = BeautifulSoup(html_main, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    md_text = markdownify(str(soup), heading_style="ATX")
    md_text = html.unescape(md_text)
    md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()
    return md_text


@serve.deployment
class LangChainLlamaStackService:
    """Ray Serve deployment for LangChain + Llama Stack document processing"""

    def __init__(self):
        print("🚀 Initializing LangChain + Llama Stack Service...")

        # Initialize Llama Stack client
        self.client = LlamaStackClient(base_url="http://localhost:8321/")

        # Initialize LangChain-compatible LLM
        self.llm = LlamaStackLLM(self.client)

        # Create processing chains
        self.summary_chain = LLMChain(llm=self.llm, prompt=summary_template)
        self.facts_chain = LLMChain(llm=self.llm, prompt=facts_template)
        self.qa_chain = LLMChain(llm=self.llm, prompt=qa_template)

        # Storage for processed documents
        self.processed_docs = {}

        print("✅ Service initialized successfully!")

    async def __call__(self, request: Request) -> Dict:
        """Handle HTTP requests to different endpoints"""
        path = request.url.path
        method = request.method

        try:
            if path == "/" and method == "GET":
                return await self._handle_status()
            elif path == "/process" and method == "POST":
                return await self._handle_process(request)
            elif path == "/ask" and method == "POST":
                return await self._handle_ask(request)
            elif path == "/summary" and method == "GET":
                return await self._handle_summary(request)
            elif path == "/facts" and method == "GET":
                return await self._handle_facts(request)
            elif path == "/docs" and method == "GET":
                return await self._handle_list_docs()
            else:
                return {
                    "error": "Not found",
                    "available_endpoints": {
                        "GET /": "Service status",
                        "POST /process": "Process document (body: {\"source\": \"url_or_path\"})",
                        "POST /ask": "Ask question (body: {\"question\": \"your_question\", \"source\": \"optional_doc_id\"})",
                        "GET /summary?source=doc_id": "Get document summary",
                        "GET /facts?source=doc_id": "Get document facts",
                        "GET /docs": "List processed documents"
                    }
                }
        except Exception as e:
            return {"error": str(e)}

    async def _handle_status(self) -> Dict:
        """Return service status"""
        return {
            "status": "healthy",
            "service": "LangChain + Llama Stack Document Processing",
            "documents_processed": len(self.processed_docs),
            "available_models": [m.identifier for m in self.client.models.list()],
            "endpoints": ["/", "/process", "/ask", "/summary", "/facts", "/docs"]
        }

    async def _handle_process(self, request: Request) -> Dict:
        """Process a document from URL or file path"""
        body = await request.json()
        source = body.get("source")

        if not source:
            return {"error": "Missing 'source' in request body"}

        try:
            # Load document
            document = load_document(source)

            # Generate summary and facts
            summary = self.summary_chain.invoke({"document": document})["text"]
            facts = self.facts_chain.invoke({"document": document})["text"]

            # Store processed document
            self.processed_docs[source] = {
                "document": document,
                "summary": summary,
                "facts": facts,
                "processed_at": time.time()
            }

            return {
                "success": True,
                "source": source,
                "document_length": len(document),
                "summary_preview": summary[:200] + "..." if len(summary) > 200 else summary,
                "facts_preview": facts[:300] + "..." if len(facts) > 300 else facts
            }

        except Exception as e:
            return {"error": f"Failed to process document: {str(e)}"}

    async def _handle_ask(self, request: Request) -> Dict:
        """Answer questions about processed documents"""
        body = await request.json()
        question = body.get("question")
        source = body.get("source")

        if not question:
            return {"error": "Missing 'question' in request body"}

        if not self.processed_docs:
            return {"error": "No documents processed yet. Use /process endpoint first."}

        try:
            # Select document
            if source and source in self.processed_docs:
                doc_data = self.processed_docs[source]
            else:
                # Use the most recent document
                doc_data = list(self.processed_docs.values())[-1]
                source = list(self.processed_docs.keys())[-1]

            # Generate answer
            answer = self.qa_chain.invoke({
                "document": doc_data["document"],
                "question": question
            })["text"]

            return {
                "question": question,
                "answer": answer,
                "source": source
            }

        except Exception as e:
            return {"error": f"Failed to answer question: {str(e)}"}

    async def _handle_summary(self, request: Request) -> Dict:
        """Get summary of a processed document"""
        source = request.query_params.get("source")

        if not self.processed_docs:
            return {"error": "No documents processed yet"}

        if source and source in self.processed_docs:
            doc_data = self.processed_docs[source]
        else:
            # Use the most recent document
            doc_data = list(self.processed_docs.values())[-1]
            source = list(self.processed_docs.keys())[-1]

        return {
            "source": source,
            "summary": doc_data["summary"]
        }

    async def _handle_facts(self, request: Request) -> Dict:
        """Get facts from a processed document"""
        source = request.query_params.get("source")

        if not self.processed_docs:
            return {"error": "No documents processed yet"}

        if source and source in self.processed_docs:
            doc_data = self.processed_docs[source]
        else:
            # Use the most recent document
            doc_data = list(self.processed_docs.values())[-1]
            source = list(self.processed_docs.keys())[-1]

        return {
            "source": source,
            "facts": doc_data["facts"]
        }

    async def _handle_list_docs(self) -> Dict:
        """List all processed documents"""
        docs_info = []
        for source, data in self.processed_docs.items():
            docs_info.append({
                "source": source,
                "document_length": len(data["document"]),
                "processed_at": data["processed_at"],
                "summary_preview": data["summary"][:100] + "..." if len(data["summary"]) > 100 else data["summary"]
            })

        return {
            "processed_documents": docs_info,
            "total_count": len(self.processed_docs)
        }


def main():
    """Main function to start the Ray Serve application"""

    # Create the application
    app = LangChainLlamaStackService.bind()

    # Deploy the application locally
    print("🚀 Starting LangChain + Llama Stack Ray Serve application...")
    serve.run(app, route_prefix="/")

    # Wait for service to initialize
    print("⏳ Waiting for service to initialize...")
    time.sleep(5)

    # Test the service
    try:
        response = requests.get("http://localhost:8000/")
        print(f"✅ Service response: {response.json()}")
        print("🎉 Service is running successfully!")
    except Exception as e:
        print(f"⚠️ Could not test service: {e}")
        print("   Service might still be starting up...")

    # Show service information
    print("\n" + "="*60)
    print("🌐 LangChain + Llama Stack Service is running on:")
    print("   http://localhost:8000/")
    print("="*60)
    print("📋 Available endpoints:")
    print("   GET  /           - Service status")
    print("   POST /process    - Process document")
    print("   POST /ask        - Ask questions")
    print("   GET  /summary    - Get document summary")
    print("   GET  /facts      - Get document facts")
    print("   GET  /docs       - List processed documents")
    print("="*60)
    print("🧪 Example requests:")
    print("   # Process a document:")
    print("   curl -X POST http://localhost:8000/process \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"source\": \"https://example.com/article\"}'")
    print("")
    print("   # Ask a question:")
    print("   curl -X POST http://localhost:8000/ask \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"question\": \"What is the main topic?\"}'")
    print("")
    print("   # Get summary:")
    print("   curl http://localhost:8000/summary")
    print("="*60)
    print("🛑 Press Ctrl+C to stop the service...")

    try:
        # Keep the service alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping service...")
        serve.shutdown()
        print("👋 Service stopped successfully!")

if __name__ == "__main__":
    main()








# import requests

# # Step 1: First, process/load the document
# process_response = requests.post(
#     "http://localhost:8000/process",
#     json={"source": "https://en.wikipedia.org/wiki/What%27s_Happening!!"}
# )
# print("Processing result:", process_response.json())

# # Step 2: Then get the facts
# facts_response = requests.get("http://localhost:8000/facts")
# print("Facts:", facts_response.json())

# # Or get facts for specific document
# facts_response = requests.get(
#     "http://localhost:8000/facts",
#     params={"source": "https://en.wikipedia.org/wiki/What%27s_Happening!!"}
# )
# print("Facts for specific doc:", facts_response.json())

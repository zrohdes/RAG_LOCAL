from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import os


class RAGChatbot:
    def __init__(self, docs_dir="./docs", model_name="llama3.2"):
        self.docs_dir = docs_dir
        self.model_name = model_name
        self.vector_store = None
        self.qa_chain = None

    def load_documents(self):
        texts = []
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(self.docs_dir, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.create_documents(texts)

    def initialize_vector_store(self, texts):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

    def setup_qa_chain(self):
        llm = OllamaLLM(model=self.model_name)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )

    def setup(self):
        texts = self.load_documents()
        self.initialize_vector_store(texts)
        self.setup_qa_chain()

    def query(self, question: str) -> str:
        if not self.qa_chain:
            raise ValueError("Chatbot not initialized. Call setup() first.")
        return self.qa_chain.run(question)


if __name__ == "__main__":
    chatbot = RAGChatbot()
    chatbot.setup()

    while True:
        question = input("\nQuestion (or 'quit'): ")
        if question.lower() == 'quit':
            break
        try:
            print(f"\nAnswer: {chatbot.query(question)}")
        except Exception as e:
            print(f"Error: {str(e)}")
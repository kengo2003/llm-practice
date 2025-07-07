import os
import sys

from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr

model_path = os.path.join("models", "Mistral-7B-Instruct-v0.3.Q3_K_M.gguf")

with SuppressStderr():
  llm = LlamaCpp(
      model_path=model_path,
      temperature=0.1,
      max_tokens=2048,
      top_p=1,
      n_ctx=2048,
      n_gpu_layers=40,  # GPU使用（Metal対応）
      f16_kv=True,  # メモリ効率化
      verbose=False,  # 詳細ログを無効化
      logits_all=False,  # ログ出力を削減
  )

documents = []
docs_path = os.path.abspath("./docs")
for filename in os.listdir(docs_path):
  path = os.path.join(docs_path, filename)
  if filename.endswith(".pdf"):
    documents.extend(PyPDFLoader(path).load())
  elif filename.endswith(".md"):
    documents.extend(UnstructuredMarkdownLoader(path).load())
  elif filename.endswith(".txt"):
    documents.extend(TextLoader(path).load())

splitter =RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=100)
split_docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs,embeddings)
retriever = vectorstore.as_retriever()

rag_chain = RetrievalQA.from_chain_type(
  llm=llm,
  retriever=retriever,
  chain_type="stuff"
)

# .pdfの内容
# query = "What does Kengo Saito do for a living?" #答え：Engineer ◯
# query = "What is Kengo Saito's favorite food?" #答え：ramen ◯
query = "Does anyone like ramen?" #答え：Kengo Saito ◯

# .mdの内容
# query = "Who likes red?" #答え：Taro Yamamoto ◯ チャンクサイズを大きくすると成功
# query = "What is Taro Yamamoto's favorite color?" #答え：red ◯

# .txtの内容
# query = "What strategy will cripple the enemy's entire defenses and open a breakthrough?"


response = rag_chain.invoke(query)
print("Query:", response["query"])
print("Response:", response["result"])


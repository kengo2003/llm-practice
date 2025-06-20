from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

docs_path = "./docs"
all_docs = []

# ドキュメントと読み込み
for filename in os.listdir(docs_path):
  if filename.endswith(".txt"):
    loader = TextLoader(os.path.join(docs_path, filename), encoding="utf-8")
    documents = loader.load()
    for doc in documents:
      doc.metadata["source"] = filename
    all_docs.extend(documents)

# チャンクに分割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(all_docs)

# モデル読み込み
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

# ベクトルDB作成
vectorstore = FAISS.from_documents(split_docs,embedding_model)
retriever = vectorstore.as_retriever()

# LLM構築
model_id = "microsoft/DialoGPT-medium" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="auto",
  torch_dtype="auto"
)

hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# RAGチェーン組む
rag_chain = RetrievalQA.from_chain_type(
  llm=llm, #応答生成エンジン
  retriever=retriever, #検索エンジン
  return_source_documents=True
)

# 実行
query = "Please tell me about strategic decision-making from this set of documents"
result = rag_chain(query)

print("\n--- 回答 ---\n")
print(result["result"])

print("\n--- 使用されたソースチャンク ---\n")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}: {doc.page_content[:80]}...")


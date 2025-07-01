from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# step1
base_model_id = "google/flan-t5-small"
# 絶対パスを使用してLoRAモデルを読み込む
lora_path = os.path.abspath("../LoRA/lora-flan/checkpoint-90")

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, lora_path)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

# step2
llm = HuggingFacePipeline(pipeline=pipe)

# step3
documents = []
docs_path = os.path.abspath("../docs")
for filename in os.listdir(docs_path):
  path = os.path.join(docs_path, filename)
  if filename.endswith(".pdf"):
    documents.extend(PyPDFLoader(path).load())
  elif filename.endswith(".md"):
    documents.extend(UnstructuredMarkdownLoader(path).load())
  elif filename.endswith(".txt"):
    documents.extend(TextLoader(path).load())

# step4
splitter =RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=100)
split_docs = splitter.split_documents(documents)

# step5
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs,embeddings)
retriever = vectorstore.as_retriever()

# step6
rag_chain = RetrievalQA.from_chain_type(
  llm=llm,
  retriever=retriever,
  chain_type="stuff"
)

# step7

# .pdfの内容
# query = "What does Kengo Saito do for a living?" #答え：Engineer ◯
# query = "What is Kengo Saito's favorite food?" #答え：ramen ◯
# query = "Does anyone like ramen?" #答え：Kengo Saito ×

# .mdの内容
# query = "Who likes red?" #答え：Taro Yamamoto ◯ チャンクサイズを大きくすると成功
# query = "What is Taro Yamamoto's favorite color?" #答え：red ◯

# .txtの内容
# query = "What strategy will cripple the enemy's entire defenses and open a breakthrough?"

# LoRA用クエリ
query = "What is the capital of Tokyo?" # ◯

response = rag_chain.invoke(query)
print("Query:", response["query"])
print("Response:", response["result"])

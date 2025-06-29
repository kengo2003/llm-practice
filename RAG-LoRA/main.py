from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


base_model_id = base_model_id = "google/flan-t5-small"
lora_path = "./lora-flan/checkpoint-1800"

# step1
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)

# LoRA適用
model = PeftModel.from_pretrained(base_model,lora_path)

# 推論パイプライン作成
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

# step2
llm = HuggingFacePipeline(pipeline=pipe)

# step3
# ドキュメント読み込み
loader = DirectoryLoader("./docs",glob="*.txt")
documents = loader.load()

# 分割
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
split_docs = splitter.split_documents(documents)

# 埋め込み
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs,embeddings)

# Retriever作成
retriever = vectorstore.as_retriever()

# step4
# LLM+Retriever統合
rag_chain = RetrievalQA.from_chain_type(
  llm=llm,
  retriever=retriever,
  chain_type="stuff"
)

# step5
# 推論実行
query = "What strategy will cripple the enemy's entire defenses and open a breakthrough?"
response = rag_chain.invoke(query)

print("==回答==")
print("query:", response["query"])
print("Response:", response["result"])
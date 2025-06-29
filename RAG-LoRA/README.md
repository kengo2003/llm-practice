# RAG × LoRA 統合

## 目的

- LoRA でファインチューニングしたモデルを RAG の生成エンジンとして活用
- LangChain を使って Retrieval と統合し、独自文書に基づく QA を実現
- 実務に耐えるドキュメント QA システムを構築

## 手順

| ステップ | 内容  
| Step1 | LoRA モデルを HuggingFace Pipeline でラップ  
| Step2 | LangChain の LLM として使えるようにする  
| Step3 | VectorStore + Retriever 構築（`docs/`配下の文書を対象）  
| Step4 | LangChain RAG 構成に LoRA モデルを組み込む  
| Step5 | 実行＆検証：ユーザーの文書に対する QA

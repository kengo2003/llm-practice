# RAG × LoRA × 他形式ファイル 統合

## 目的

- LoRA でファインチューニングしたモデルを使用する
- PDF や Markdown 資料を学習データ化する
- LangChain による RAG を実装する

## 手順

| step | 内容  
| Step1 | LoRA モデルの読み込みと Pipeline 化  
| Step2 | LangChain で使える LLM 形式に変換  
| Step3 | 文書読み込み（PDF / Markdown / txt 対応）  
| Step4 | トークナイザを意識したチャンク分割  
| Step5 | ベクトル化と Retriever 構築  
| Step6 | RAG チェーンを作成（LLM + Retriever）  
| Step7 | 実行

<div align="center">
🌾 AgroMind

Climate-Smart Crop Advisor

Fine-tuning Gemma 2B-IT with QLoRA + Retrieval-Augmented Generation for India's Smallholder Farmers

Show Image
Show Image
Show Image
Show Image
Show Image
Show Image

Built for the Generative AI & LLMs course — BML Munjal University

</div>

📖 Overview

AgroMind is a domain-specialized LLM system that recommends the most suitable crop to grow given a farmer's soil (N-P-K) and climate (temperature, humidity, pH, rainfall) readings — along with a plain-language reasoning trace and a risk assessment.

It's built on Gemma 2B-IT, fine-tuned with QLoRA on the Kaggle Crop Recommendation dataset, and augmented with a RAG pipeline (ChromaDB) for semantic few-shot grounding. Every inference is logged to a SQLite audit trail for monitoring and future retraining.

The entire pipeline runs on a single 6GB consumer GPU (RTX 3060), making it realistic to deploy for agricultural extension use in low-resource settings.


Result: Crop recommendation accuracy improved from 52% → 96%, and hallucination rate dropped from 36% → 2%, compared to the zero-shot base model.




✨ Key Features

🧠 QLoRA Fine-Tuning4-bit NF4 quantized Gemma 2B-IT, LoRA adapters on all attention + MLP projections — trains on 6GB VRAM🔍 RAG RetrievalChromaDB + all-MiniLM-L6-v2 embeddings retrieve top-3 agronomically similar examples at inference time🗃️ Dual-Storage ArchitectureChromaDB (vector) for semantic retrieval + SQLite (relational) for structured query logging📊 Rigorous EvaluationROUGE-1/2/L, BLEU, and a custom Crop Accuracy metric, benchmarked against two baselines🧪 Error & Hallucination Analysis3-type error taxonomy with adversarial/edge-case stress testing🇮🇳 India-First DesignCovers 22 crops across agro-climatic zones — Rajasthan, Punjab, and the northeast⚡ Edge-Deployable~34MB LoRA adapter; mergeable and GGUF-quantizable for CPU-only inference via llama.cpp


🏗️ Architecture

                         ┌─────────────────────────┐
                         │   Farmer Input (N,P,K,   │
                         │  temp, humidity, pH, rain)│
                         └────────────┬────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
        ┌───────────────────────┐         ┌───────────────────────────┐
        │   ChromaDB Retrieval   │         │  Gemma 2B-IT (4-bit NF4)  │
        │  (top-3 similar crop   │◄───────►│   + QLoRA Adapter (r=16)  │
        │   profiles, cosine)    │  RAG    │      ~8.4M trainable      │
        └───────────────────────┘  ctx     │        params (0.34%)     │
                     │                     └─────────────┬─────────────┘
                     │                                   │
                     └──────────────┬────────────────────┘
                                    ▼
                     ┌───────────────────────────────┐
                     │   Crop + Reasoning + Risk      │
                     │   Level Recommendation         │
                     └───────────────┬───────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │  SQLite Audit Log (latency,    │
                     │  context used, output, ts)     │
                     └───────────────────────────────┘


📊 Dataset

Source: Kaggle Crop Recommendation Dataset


2,200 samples · 22 Indian crop classes · perfectly balanced (100 samples/class)
7 agronomic features: N, P, K (ppm), temperature (°C), humidity (%), soil pH, rainfall (mm)
Converted to instruction-tuning pairs with a templated formatter that generates natural-language reasoning + heuristic risk labels (Low / Medium / High, based on rainfall, temperature, and pH stress flags)


SplitSamples%Train1,54070%Validation33015%Test33015%

Split via stratified sampling, random_state=42, saved as data/processed/{train,val,test}.jsonl.


🔧 Fine-Tuning Configuration

Base model: google/gemma-2b-it — chosen for its strong instruction-tuned zero-shot baseline, its fit within 6GB VRAM at 4-bit precision, and native chat-template support.

QLoRA Setup

ParameterValueRationaleRank (r)16Balances adapter quality vs. VRAM budgetLoRA alpha32Standard alpha = 2×r scalingTarget modulesq,k,v,o,gate,up,down_projFull attention + MLP coverageLoRA dropout0.05Mild regularizationQuantization4-bit NF4 + double quant~16GB → ~5GB VRAMTrainable params~8.4M / 2.5B (0.34%)Efficient adaptation

Training Arguments

ArgumentValueEpochs3Batch size1 (grad. accumulation ×4, effective batch = 4)Max sequence length256Learning rate2e-4, cosine schedule, 5% warmupOptimizerpaged_adamw_8bit

Training and validation loss converge smoothly over 300 steps with no signs of overfitting.


🗃️ Data Storage

ChromaDB (Vector DB) — persisted to chromadb_store/


Stores sentence embeddings (384-dim, all-MiniLM-L6-v2) of training input-output pairs / crop profiles
Cosine-similarity retrieval, top-3 nearest neighbors injected as context at inference


SQLite (Relational Log) — persisted to agromind_logs.db


Table query_logs: id, timestamp, input_text, context, output, model_type
Powers audit trails, latency tracking, and feedback collection for future fine-tuning rounds



📈 Results

Three configurations were evaluated on an identical 50-sample test subset:


Baseline 1 — Zero-shot Gemma 2B-IT: minimal system prompt, no retrieval, no fine-tuning
Baseline 2 — RAG-only: base model + ChromaDB top-3 retrieval, no fine-tuning
AgroMind (FT + RAG): QLoRA adapter + ChromaDB retrieval — the full system


MetricBaseline 1 (Zero-shot)Baseline 2 (RAG-only)AgroMind (FT + RAG)Δ vs. B1ROUGE-10.3120.4810.763+145%ROUGE-20.1180.2430.581+392%ROUGE-L0.2890.4460.741+156%BLEU0.0610.1470.412+575%Crop Accuracy52.0%74.0%96.0%+85%

Error Taxonomy (out of 50 test samples)

Error TypeBaseline 1Baseline 2AgroMindHallucination (non-existent crop)1881Wrong Crop (valid crop, wrong pick)1472Missing Reasoning651Correct (no error)123046

Hallucination rate: 36% → 2% (−94%). Without domain anchoring, the base Gemma model defaults to globally common crops (e.g., wheat, barley) that fall outside the 22-crop vocabulary. Fine-tuning enforces distributional concentration on the closed crop set, while RAG retrieval further anchors generations with in-distribution examples.

Adversarial edge cases (conflicting inputs, extreme stress values, ambiguous midpoint conditions) were also tested — AgroMind correctly flags high-risk conditions and hedges appropriately on ambiguous inputs rather than confidently guessing wrong.


🗂️ Repository Structure

agromind/
├── agromind_finetune.ipynb        # End-to-end pipeline notebook
├── data/
│   ├── Crop_recommendation.csv    # Raw Kaggle dataset
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
├── agromind_adapter/               # Saved QLoRA adapter weights (~34MB)
├── agromind_merged/                 # (Optional) merged full model
├── chromadb_store/                  # Persisted vector DB
├── agromind_logs.db                 # SQLite query audit log
├── evaluation_results.csv           # Metric comparison table
├── all_model_outputs.json           # Raw generations from all 3 configs
└── README.md


🚀 Getting Started

Requirements


Python 3.11
CUDA-capable GPU with ≥6GB VRAM (tested on RTX 3060)
Hugging Face account with access to google/gemma-2b-it


Installation

bashgit clone https://github.com/<your-username>/agromind.git
cd agromind

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 peft==0.10.0 trl==0.8.6 accelerate==1.6.0 datasets==2.18.0
pip install bitsandbytes==0.46.1 --prefer-binary
pip install chromadb sentence-transformers langchain langchain-community
pip install evaluate rouge-score sacrebleu
pip install pandas==2.0.3 numpy==1.26.4 scikit-learn

Hugging Face Authentication

Set your HF token as an environment variable rather than hardcoding it in notebooks or scripts:

bashexport HF_TOKEN="your_token_here"       # macOS/Linux
setx HF_TOKEN "your_token_here"         # Windows

pythonfrom huggingface_hub import login
import os
login(token=os.environ["HF_TOKEN"])


⚠️ Security note: Never commit HF tokens, API keys, or credentials to version control. Rotate any token that has ever been pasted into a script or notebook.



Running the Pipeline

Open agromind_finetune.ipynb and run sequentially:


Dataset prep — load, preprocess, and split the Kaggle dataset
Baseline 1 — zero-shot Gemma 2B-IT generation
QLoRA fine-tuning — trains and saves the adapter to agromind_adapter/
RAG setup — builds the ChromaDB knowledge base and SQLite logger
Baseline 2 & AgroMind inference — generate outputs on the test subset
Evaluation — computes ROUGE/BLEU/Crop Accuracy and saves comparison tables
Error analysis — hallucination taxonomy and adversarial test cases


Expected fine-tuning time: ~2–3 hours for 3 epochs on an RTX 3060 6GB.


🌍 Real-World Applicability


Agro-climatic coverage — crops spanning Rajasthan (bajra, cotton), Punjab (wheat, rice), and northeastern India (jute)
Resource-constrained deployment — the ~34MB adapter runs on a laptop GPU; the merged model can be GGUF-quantized for CPU-only inference via llama.cpp
Interpretable outputs — every recommendation includes NPK justification, optimal-range reasoning, and an explicit risk classification, built for farmer and extension-officer trust
Auditability — SQLite logging enables drift monitoring and feedback collection for continuous fine-tuning



🔮 Limitations & Future Work


Model scale — Gemma 2B-IT is small; scaling to Llama-3-8B with similar QLoRA setup would likely push ROUGE-2 above 0.70
Static RAG — current retrieval draws on fixed training examples; a future version should ingest real-time IMD weather API and soil health card data
Multilingual access — adding a Hindi/Punjabi translation layer would substantially widen farmer accessibility



📚 References


Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.
Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
Google DeepMind (2024). Gemma: Open Models Based on Gemini Research and Technology. arXiv:2403.08295.
Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
Kaggle Crop Recommendation Dataset
Hugging Face PEFT Library
ChromaDB Docs



📄 License

This project is released under the MIT License. See LICENSE for details.

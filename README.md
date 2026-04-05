# CrossIndexRAG

A multimodal Retrieval-Augmented Generation (RAG) system that retrieves across **text, image, and table** data in a single unified pipeline — with cross-encoder reranking, score fusion, and hallucination control through modality agreement.

---

## What it does

Most RAG systems only search text. CrossIndexRAG searches all three modalities at once and combines the results intelligently:

1. **Hybrid Retrieval** — pulls candidates from text, image, and table indexes simultaneously
2. **Cross-Encoder Reranking** — a second-stage model rescores every candidate against the query for higher precision
3. **Score Fusion** — combines embedding similarity and cross-encoder scores using weighted min-max normalisation
4. **Modality Agreement** — flags a response as "safe" only when multiple independent modalities (e.g. both a text document and a table row) agree — reducing hallucination risk

---

## Architecture

```
CrossIndexRAG/
├── app/                    # Entry point — interactive query loop
├── embedders/              # Per-modality embedding models
│   ├── text_embedder.py    # multi-qa-mpnet-base-dot-v1
│   ├── image_embedder.py   # CLIP (openai/clip-vit-base-patch32)
│   └── table_embedder.py   # all-MiniLM-L6-v2
├── vector_db/              # ChromaDB collections per modality
├── retrievers/             # Search logic per modality
├── reranker/               # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
├── fusion/                 # Score fusion + modality agreement
├── evaluation/             # Precision@K, Recall@K, nDCG, MRR metrics
├── configs/                # Central config (models, hyperparameters)
├── utils/                  # Logger, document schema
└── ingest_data.py          # One-time data ingestion script
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Eshal-ven/CrossIndexRAG.git
cd CrossIndexRAG
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your data**

Place your files into the correct subfolders:
```
data/
  text_docs/    →  .txt files
  image_docs/   →  .jpg / .png files
  table_docs/   →  .csv files (one row = one document)
```

**4. Ingest data into ChromaDB**
```bash
python ingest_data.py
```

**5. Run the system**
```bash
python app/hybridRetrievalDemo.py
```

You will see an interactive prompt:
```
Hybrid Retrieval System Ready (type 'exit' to quit)

Query> who are the people in the dataset
```

---

## Example Output

```
Query: who are the people in the dataset
Time: 1.682s  | Safe: True

Rank 1 | Modality: image  | ID: image_sample
Scores -> embed: 0.2289, ce: -11.1993, final: 0.8756

Rank 2 | Modality: text   | ID: text_hello
Scores -> embed: 0.2772, ce: -11.4709, final: 0.3809

Rank 3 | Modality: table  | ID: table_people_5
Preview: Name: Ayesha Malik | Age: 27 | City: Multan | Profession: Teacher
Scores -> embed: 0.2158, ce: -11.4410, final: 0.2909
```

**Safe: True** means results came from at least 2 distinct modalities — the hallucination control is working.

---

## Evaluation

Run the built-in benchmark to measure retrieval quality:

```python
from evaluation.evaluator import Evaluator

benchmark = [
    {"query": "your query here", "relevant_ids": ["text_doc1", "table_row2"]}
]

evaluator = Evaluator(app)
report = evaluator.run(benchmark, k=5)
evaluator.print_report(report)
```

Metrics reported: `Precision@K`, `Recall@K`, `nDCG@K`, `MRR`, `Modality Coverage`, `Safe Rate`

---

## Tech Stack

| Component | Model / Library |
|---|---|
| Text embedding | sentence-transformers/multi-qa-mpnet-base-dot-v1 |
| Image embedding | openai/clip-vit-base-patch32 |
| Table embedding | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector DB | ChromaDB |
| Framework | Python, HuggingFace, PyTorch |

---

## Requirements

```
chromadb
sentence-transformers
transformers
torch
Pillow
numpy
scikit-learn
tqdm
huggingface-hub
```

---

## Author

**Eshal Fatima** — AI Engineer
AI Researcher @ Swan Labs | Python Developer Intern @ Arch Technologies

[LinkedIn](https://www.linkedin.com/in/eshal-fatima-) | [GitHub](https://github.com/Eshal-ven) | [Kaggle](https://www.kaggle.com/eashelfatima)# CrossIndexRAG

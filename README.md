### **Introduction**

**1. Background**
The Indian Parliament (Lok Sabha) generates an overwhelming volume of dense, legally complex, and multilingual legislative data, including bills, parliamentary debates, and committee reports. Historically, navigating this data to extract specific legal facts—such as financial penalties, statutory clauses, or exact jail terms—has been a manual and time-consuming process for researchers and legal professionals. 
In recent years, Artificial Intelligence, specifically **Large Language Models (LLMs)**, has been introduced to automate information retrieval. To ground these models in factual data, the industry standard is **Retrieval-Augmented Generation (RAG)**. Key terms in this domain include **Dense Embeddings** (capturing the semantic "vibe" of text), **Sparse Keyword Arrays** (exact keyword matching like BM25), and **Cross-Encoders** (models that calculate exact relevance scores between a query and a document using cross-attention mechanisms).

**2. Existing Evidence**
Current literature and industry applications heavily rely on single-stage RAG pipelines using standard Bi-Encoders. While effective for general knowledge, existing implementations struggle in legal contexts. Standard vector searches frequently miss exact numerical data (e.g., a "₹5,000 fine") buried in long texts. Furthermore, generative models are prone to "hallucinations"—confidently fabricating laws or penalties when the retrieval engine fails to provide the exact context. Additionally, the majority of state-of-the-art embedding models and rerankers are highly optimized for English and Chinese datasets, showing significant degradation when processing code-mixed texts or native Indian languages.

**3. Research Gap**
There is a critical gap in applying standard RAG architectures to the Lok Sabha dataset. Specifically, three major problems remain unsolved in basic implementations:
* **The Multilingual Reranking Blindspot:** Standard cross-encoders (like `bge-reranker-base`) lack the tokenization rules for Devanagari and Dravidian scripts. Consequently, they treat native Indian languages as noise, artificially scoring them low and mathematically discarding critical legal documents before the LLM can read them.
* **Precision vs. Hallucination:** Single-stage retrieval often feeds the LLM too much irrelevant context, leading to the "Lost in the Middle" syndrome where the AI guesses missing legal caveats (like "whichever is higher") instead of admitting ignorance. 
* **Hardware and Ingestion Limits:** Processing and fusing massive, multi-modal vector embeddings (241,000+ chunks) reliably causes Out-of-Memory (OOM) crashes on standard consumer hardware, preventing local deployment.

**4. Objective**
The objective of this project is to architect, deploy, and evaluate a hallucination-resistant, hardware-optimized **Two-Stage Hybrid RAG System** for Lok Sabha data. Specifically, this project plans to accomplish:
* The deployment of a localized, low-RAM constrained vector database (Qdrant) capable of handling massive ingestions.
* The implementation of Reciprocal Rank Fusion (RRF) to perfectly balance dense semantic search (`BAAI/bge-m3`) with sparse exact-keyword matching (`BM25`).
* The integration of a Cross-Encoder reranking layer to aggressively filter context, forcing the cloud-based LLM (`gemini-2.5-flash`) to achieve near-zero hallucination rates.
* The development of a Pre-Retrieval / Pre-Ingestion Data Normalization layer to seamlessly handle Indian language queries (Hindi, Kannada, Tamil) without triggering the latency bottlenecks of massive multi-lingual rerankers.

**5. Scope**
The scope of this research is constrained by a few practical and technical factors:
* **Dataset Constraints:** The project relies on a static, pre-extracted JSON corpus of approximately 241,000 chunked parliamentary documents. It does not actively scrape live, daily updates from the Lok Sabha website.
* **Hardware Constraints:** The architecture is specifically bounded by consumer-grade laptop RAM and CPU limits, meaning computational complexity must be strictly managed (e.g., opting against $O(N^2)$ multi-lingual rerankers in favor of API translation layers to maintain sub-minute query latency).
* **API Constraints:** The generation and translation layers rely on cloud-based LLMs subject to standard rate limits, meaning large-scale batch normalization must be throttled or transitioned to local open-source models for massive offline runs.

**6. References**
A Retrieval-augmented Generation Framework with Retriever and Generator Modules for Enhancing Factual Consistency: https://www.researchgate.net/publication/393590507_A_Retrieval-augmented_Generation_Framework_with_Retriever_and_Generator_Modules_for_Enhancing_Factual_Consistency
To Enhance Graph-Based Retrieval-Augmented Generation (RAG) with Robust Retrieval Techniques: https://ieeexplore.ieee.org/document/10871140
TKG-RAG: A Retrieval-Augmented Generation Framework with Text-chunk Knowledge Graph: https://ieeexplore.ieee.org/document/10877117
Knowledge Graph-Guided Retrieval Augmented Generation: https://aclanthology.org/2025.naacl-long.449.pdf
KemenkeuGPT: Leveraging a Large Language Model on Indonesia's Government Financial Data and Regulations to Enhance Decision Making: https://arxiv.org/abs/2407.21459
Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Task: https://arxiv.org/abs/2504.03616
Retrieval-augmented generation in multilingual settings:https://arxiv.org/html/2407.01463v1
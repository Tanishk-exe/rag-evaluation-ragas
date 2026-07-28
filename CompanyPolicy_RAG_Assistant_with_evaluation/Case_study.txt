RAG Evaluation Framework
Case Study: Comparative Evaluation of RAG Pipeline Configurations using RAGAS
Abstract

This case study evaluates the performance of different Retrieval-Augmented Generation (RAG) configurations using the RAGAS evaluation framework. The objective is to identify how changing the Large Language Model (LLM) and embedding model impacts the quality of generated answers. Performance is measured using four RAGAS metrics: Faithfulness, Context Recall, Context Precision, and Answer Relevancy.

1. Objective

The primary goal of this study is to benchmark different RAG configurations and determine which setup produces the most reliable responses for an HR Policy knowledge base.

The experiments focus on:

Comparing multiple LLMs
Evaluating the impact of embedding models
Measuring retrieval and generation quality using RAGAS
2. Experimental Setup
Component	Baseline Configuration
Document	HR Policy PDF
Evaluation Framework	RAGAS
Vector Store	ChromaDB
Retriever	Similarity Search
Chunk Size	500
Chunk Overlap	50
Top-K	4
3. Evaluation Metrics
Metric	Description
Faithfulness	Measures whether the generated answer is supported by the retrieved context.
Context Recall	Measures whether all relevant information was successfully retrieved.
Context Precision	Measures how much of the retrieved context is actually relevant.
Answer Relevancy	Measures how well the answer addresses the user's question.
4. Experimental Results
Experiment	Faithfulness	Context Recall	Context Precision	Answer Relevancy
Baseline	0.90	0.83	0.61	0.78
Llama-70B	0.82	0.83	0.50	0.91
GPT-120B	0.80	0.80	0.75	1.00
Qwen-27B	0.64	0.83	0.75	1.00
BGE Small Embedding	0.95	0.83	0.89	0.74
5. Analysis
Experiment 1 – Baseline

The baseline configuration achieved a Faithfulness score of 0.90, indicating that most generated answers were well-grounded in the retrieved context. However, the Context Precision of 0.61 suggests that the retriever still returned some irrelevant information.

Experiment 2 – Llama-70B

Replacing the baseline LLM with Llama-70B resulted in a slight decrease in Faithfulness (0.82) and Context Precision (0.50). However, Answer Relevancy increased to 0.91, indicating that the model produced answers that were more directly aligned with user queries, although not always fully supported by the retrieved documents.

Experiment 3 – GPT-120B

GPT-120B achieved the highest Answer Relevancy score (1.00), demonstrating excellent ability to answer the user's question. Additionally, Context Precision improved to 0.75, indicating more relevant retrieved information compared to the baseline.

Experiment 4 – Qwen-27B

Qwen-27B also achieved a perfect Answer Relevancy score (1.00), but its Faithfulness dropped significantly to 0.64. This suggests that while the answers were highly relevant, they were less consistently grounded in the retrieved context and may contain unsupported information.

Experiment 5 – BGE Small Embedding

Replacing the embedding model with BGE Small produced the best retrieval performance among all experiments.

Compared to the baseline:

Faithfulness improved from 0.90 → 0.95
Context Precision improved from 0.61 → 0.89
Context Recall remained unchanged at 0.83

These results indicate that improving the embedding model had a stronger positive effect on retrieval quality than changing the LLM alone.

6. Key Observations
Best Faithfulness

BGE Small Embedding (0.95)

The embedding model provided the most reliable grounding between retrieved context and generated answers.

Best Context Precision

BGE Small Embedding (0.89)

The retrieved documents contained substantially less irrelevant information.

Best Context Recall

Baseline, Llama-70B, Qwen-27B, and BGE Small all achieved 0.83, indicating comparable retrieval coverage.

Best Answer Relevancy

GPT-120B and Qwen-27B achieved a perfect 1.00, demonstrating excellent response relevance.

7. Comparative Summary
Category	Best Performer	Score
Faithfulness	BGE Small Embedding	0.95
Context Recall	Multiple Models	0.83
Context Precision	BGE Small Embedding	0.89
Answer Relevancy	GPT-120B / Qwen-27B	1.00
8. Conclusion

This evaluation demonstrates that both the language model and embedding model significantly influence the overall performance of a Retrieval-Augmented Generation system.

Among the evaluated configurations:

BGE Small Embedding produced the highest retrieval quality, leading to improvements in both Faithfulness and Context Precision.
GPT-120B and Qwen-27B generated the most relevant answers, although Qwen showed reduced Faithfulness, indicating a higher tendency to generate responses not fully supported by the retrieved context.
The baseline configuration provided a balanced trade-off between retrieval quality and answer grounding.

These findings suggest that optimizing the retrieval pipeline—particularly the embedding model—can have a substantial impact on the overall effectiveness of a RAG system.

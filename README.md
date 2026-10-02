# COSC2669-WIL-Group45

Group ID: 45

Group Name: The WIL-ing Project

## Team Members
1. Aindrila Das Prapti
2. Chethan Shankar
3. Ganesh Prabhakarareddy
4. Rickson Prajwal Pereira
5. Santhini Chandrasekar

## Project ##
Victorian Rental & Tenancy RAG Assistant
Test-Driven Retrieval-Augmented Generation (RAG) for Victorian rental and tenancy information.


Project Overview

This project develops a Retrieval-Augmented Generation (RAG) assistant for Victorian residential rental and tenancy information. The system uses a focused Consumer Affairs Victoria tenancy knowledge base, hybrid information retrieval, and a locally hosted language model to generate answers supported by retrieved tenancy sources.

End-to-end pipeline: Data Collection → Data Preparation → Indexing → Retrieval → Hybrid RAG → LLM Answer Generation → Evaluation → Streamlit Interface.

1. Data and Knowledge Base

The knowledge base contains Victorian tenancy information covering bond amounts and payments, condition reports, minimum rental standards, heating requirements, notices to vacate, rent increases, rent payment rules, repairs and urgent repairs, and rental agreement/signing rights.

The source material was prepared as smaller retrievable chunks with unique document identifiers. The prepared tenancy collection is stored under quantitative_eval/data/tenancy_collection/, with additional data under quantitative_eval/data/.

2. Retrieval Development

Two retrieval approaches were implemented with Pyserini. BM25 provides sparse lexical retrieval based on term matching. Dense retrieval uses the castorini/tct_colbert-v2-hnp-msmarco encoder and a FAISS index to retrieve semantically related tenancy passages.

The final system combines both approaches. Candidate scores are normalised and combined using a 50% BM25 and 50% dense retrieval score. Topic and evidence matching are then used to prioritise passages that directly address the user's tenancy question.

3. RAG Development and Integration

The main RAG implementation is in quantitative_eval/src/rag/chatbot.py. A user question is sent to both BM25 and dense retrieval. The retrieved candidates are combined and ranked, relevant evidence is selected, and a prompt is constructed from that evidence.

Answer generation is performed locally through Ollama using llama3.2:3b. The language model is given the retrieved tenancy evidence so that its response can be grounded in the project's knowledge base rather than generated without supporting context.

4. Streamlit Prototype

A browser-based prototype was integrated in quantitative_eval/app.py using Streamlit. Users can enter a tenancy question, receive the generated answer, and inspect the supporting retrieved sources. The interface also makes clear that the prototype provides information from retrieved tenancy sources and is not legal advice.

Run the application from quantitative_eval with:

python -m streamlit run app.py

5. Retrieval Evaluation

Retrieval effectiveness was evaluated using NDCG@1, NDCG@3 and NDCG@5 for known, inferred, and overall question sets.


The results show different strengths for lexical and semantic retrieval. BM25 performed strongly on known questions, while dense retrieval was particularly effective for inferred questions. This supports the use of both retrieval approaches in the final hybrid pipeline.

6. End-to-End Testing

The complete RAG pipeline was tested on 20 evaluation questions: 8 known questions (K01–K08) and 12 inferred/scenario questions (I01–I12).

Each test executed the full flow: Question → BM25 retrieval → Dense FAISS retrieval → Hybrid ranking → Evidence selection → Prompt construction → Ollama answer generation → Answer and source recording.

Question Set    Retriever       NDCG@1   NDCG@3   NDCG@5
Known            BM25           0.6250   0.6555   0.7632

Known           Dense FAISS     0.3750   0.6655   0.6655

Inferred         BM25           0.3333   0.4422   0.4781

Inferred        Dense FAISS     0.6667   0.6870   0.6870

Overall         BM25            0.4500   0.5275   0.5921

Overall         Dense FAISS     0.5500   0.6784   0.6784

The final evaluation completed all 20 questions. No generated answer was empty, and every result contained five retrieved evidence records. The generated results are stored in data/evaluation/rag_results/rag_answers.jsonl, with final answer-level evaluation information stored in data/evaluation/answer_evaluation_final.csv.

7. Topics Tested

Maximum bond amounts

Condition report deadlines

Heating and minimum rental standards

Notices to vacate

Rent increase notice periods

Rent paid in advance

Urgent and non-urgent repairs

Cooling-off periods

Rental payment rules

Rental agreement and signing rights


8. Technologies Used

Python, Pyserini, BM25, FAISS, TCT-ColBERT query encoding, Retrieval-Augmented Generation (RAG), Ollama, Llama 3.2 3B, Streamlit, Git and GitHub.

9. Current Status

The end-to-end prototype has been implemented, integrated and tested. Completed components include knowledge-base preparation, BM25 retrieval, dense FAISS retrieval, retrieval evaluation, hybrid retrieval, evidence selection, prompt construction, local LLM integration, automated RAG evaluation, answer-level evaluation, the Streamlit browser interface, and final integration testing.

The prototype demonstrates a complete RAG workflow for retrieving and presenting Victorian residential tenancy information with supporting source evidence. Testing also identified that individual responses can still be improved through future refinement of evidence selection and answer generation.

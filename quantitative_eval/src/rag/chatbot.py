import json
from pathlib import Path

import requests
from pyserini.search import LuceneSearcher, FaissSearcher


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BM25_INDEX = "target/indexes/tenancy-bm25"
DENSE_INDEX = "target/indexes/tenancy-tct_colbert-v2-hnp-msmarco-faiss"
QUERY_ENCODER = "castorini/tct_colbert-v2-hnp-msmarco"
COLLECTION = Path("data/tenancy_collection/collection.jsonl")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

TOP_K = 5
CANDIDATE_K = 10


# --------------------------------------------------
# Load source chunks
# --------------------------------------------------

def load_documents():
    documents = {}

    with COLLECTION.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            documents[item["id"]] = item["contents"]

    return documents


# --------------------------------------------------
# BM25 retrieval
# --------------------------------------------------

def bm25_search(searcher, query, k=CANDIDATE_K):
    results = []

    for hit in searcher.search(query, k):
        doc = searcher.doc(hit.docid)

        if doc is None or not doc.raw():
            continue

        item = json.loads(doc.raw())

        results.append({
            "id": item["id"],
            "score": float(hit.score)
        })

    return results


# --------------------------------------------------
# Dense retrieval
# --------------------------------------------------

def dense_search(searcher, query, k=CANDIDATE_K):
    return [
        {
            "id": hit.docid,
            "score": float(hit.score)
        }
        for hit in searcher.search(query, k)
    ]


# --------------------------------------------------
# Score normalization
# --------------------------------------------------

def normalize(results):
    if not results:
        return {}

    scores = [x["score"] for x in results]

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return {x["id"]: 0.0 for x in results}

    return {
        x["id"]: (x["score"] - minimum) / (maximum - minimum)
        for x in results
    }


# --------------------------------------------------
# E7.2 topic-aware evidence selection
# --------------------------------------------------

def select_evidence(query, candidates, documents):

    stop_words = {
        "a", "an", "the", "is", "are", "am", "can", "i", "my",
        "me", "to", "of", "for", "in", "on", "and", "or", "do",
        "does", "did", "what", "when", "how", "why", "if", "with",
        "just", "it", "this", "that", "they", "their", "after",
        "before", "have", "has", "had", "will", "would", "could"
    }

    query_terms = {
        term.strip(".,?!:;()[]{}\"'").lower()
        for term in query.split()
    }

    query_terms -= stop_words

    topic_groups = {

        "payment": {
            "payment", "paying", "pay", "fee", "fees",
            "charge", "charges", "method", "methods"
        },

        "bond": {
            "bond", "bonds"
        },

        "notice": {
            "notice", "vacate", "evict", "eviction",
            "kick", "terminate", "termination"
        },

        "repair": {
            "repair", "repairs", "fix", "fixed", "fault",
            "damage", "urgent", "non-urgent"
        },

        "lease": {
            "lease", "agreement", "term", "terms",
            "contract", "sign", "signed"
        },

        "rent_increase": {
            "rent", "increase", "raise", "warning",
            "notice", "days"
        }
    }

    query_topics = set()

    for topic, terms in topic_groups.items():
        if query_terms & terms:
            query_topics.add(topic)

    selected = []

    for item in candidates:

        text = documents.get(item["id"], "").lower()

        text_terms = {
            term.strip(".,?!:;()[]{}\"'")
            for term in text.split()
        }

        overlap = len(query_terms & text_terms)

        topic_score = 0

        for topic in query_topics:
            topic_terms = topic_groups[topic]

            if text_terms & topic_terms:
                topic_score += 1

        item = item.copy()

        item["evidence_score"] = overlap
        item["topic_score"] = topic_score

        selected.append(item)

    selected.sort(
        key=lambda x: (
            x["topic_score"],
            x["evidence_score"],
            x["score"]
        ),
        reverse=True
    )

    return selected[:TOP_K]


# --------------------------------------------------
# Hybrid retrieval
# --------------------------------------------------

def hybrid_search(bm25_searcher, dense_searcher, query, documents):

    bm25 = normalize(
        bm25_search(bm25_searcher, query)
    )

    dense = normalize(
        dense_search(dense_searcher, query)
    )

    all_ids = set(bm25) | set(dense)

    combined = []

    for doc_id in all_ids:

        score = (
            0.5 * bm25.get(doc_id, 0.0)
            + 0.5 * dense.get(doc_id, 0.0)
        )

        combined.append({
            "id": doc_id,
            "score": score
        })

    combined.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return select_evidence(
        query,
        combined,
        documents
    )


# --------------------------------------------------
# E7.2 structured evidence prompt
# --------------------------------------------------

def build_prompt(question, retrieved, documents):

    context_parts = []

    for item in retrieved:

        doc_id = item["id"]

        context_parts.append(
            f"SOURCE: {doc_id}\n"
            f"{documents.get(doc_id, '')}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are extracting evidence to answer the EXACT user question.

Do NOT answer with general information from the retrieved sources.

First determine exactly what the user is asking.

Then identify the specific statement in the supplied evidence that answers that question.

RULE:

- State only the rule that directly answers the exact question.

- Do not substitute another rule from the same source.

- If the source contains several rules, select only the relevant one.

- Apply the rule to the user's specific situation when the question describes a scenario.

EXCEPTIONS:

- Include only exceptions that directly affect the answer to the question.

- Do not include unrelated exceptions from the same source.

- If none directly apply, write "None stated".

CONDITIONS:

- Include only conditions necessary to understand or apply the answer.

- Do not add conditions from unrelated rules.

- If none directly apply, write "None stated".

NUMBERS/DATES:

- Include only numbers or dates directly relevant to the answer.

- Preserve them exactly.

SOURCE:

- Give the source ID that directly supports the RULE.

- The source must support the specific answer, not merely the general topic.

IMPORTANT:

- Use ONLY the supplied evidence.

- Do not use outside knowledge.

- Do not infer facts that are not stated.

- Preserve can/cannot/must/may and other rule directions exactly.

- Never reverse a rule.

- Do not combine unrelated rules.

- Do not explain your reasoning.

- Do not write a normal conversational answer.

- Output only the requested fields.

SUPPLIED CONTEXT:

{context}

USER QUESTION:

{question}

STRUCTURED EVIDENCE:
"""


# --------------------------------------------------
# Generate answer with Ollama
# --------------------------------------------------

def generate_answer(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()


# --------------------------------------------------
# Main chatbot
# --------------------------------------------------

def main():

    print("Loading documents...")
    documents = load_documents()

    print("Loading BM25 index...")
    bm25_searcher = LuceneSearcher(BM25_INDEX)

    print("Loading dense index...")
    dense_searcher = FaissSearcher(
        DENSE_INDEX,
        QUERY_ENCODER
    )

    print("\nVictorian Tenancy RAG Chatbot — E7.2")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        retrieved = hybrid_search(
            bm25_searcher,
            dense_searcher,
            question,
            documents
        )

        prompt = build_prompt(
            question,
            retrieved,
            documents
        )

        answer = generate_answer(prompt)

        print("\nAssistant:")
        print(answer)

        print("\nRetrieved sources:")

        for item in retrieved:
            print(
                f"- {item['id']} "
                f"(hybrid={item['score']:.3f}, "
                f"topic={item['topic_score']}, "
                f"evidence={item['evidence_score']})"
            )

        print()


if __name__ == "__main__":
    main()

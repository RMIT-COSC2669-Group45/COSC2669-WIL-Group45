import json
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher

INDEX = "target/indexes/tenancy-bm25"
QUESTIONS = Path("data/evaluation/questions.jsonl")
OUTPUT = Path("data/evaluation/runs/tenancy-bm25.txt")

TOP_K = 5

def main():
    searcher = LuceneSearcher(INDEX)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with QUESTIONS.open("r", encoding="utf-8") as f, OUTPUT.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            query_id = item["query_id"]
            query = item["query"]

            hits = searcher.search(query, TOP_K)

            for rank, hit in enumerate(hits, start=1):
                doc = searcher.doc(hit.docid)
                raw = json.loads(doc.raw())
                doc_id = raw["id"]

                out.write(
                    f"{query_id} Q0 {doc_id} {rank} {hit.score:.6f} tenancy-bm25\n"
                )

            print(f"{query_id}: {len(hits)} results")

    print(f"\nRun saved to: {OUTPUT}")

if __name__ == "__main__":
    main()

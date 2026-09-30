import json
from pathlib import Path
from pyserini.search import FaissSearcher

INDEX = "target/indexes/tenancy-tct_colbert-v2-hnp-msmarco-faiss"
QUERY_ENCODER = "castorini/tct_colbert-v2-hnp-msmarco"

QUESTIONS = Path("data/evaluation/questions.jsonl")
DOCIDS = Path("target/embeddings/tenancy-tct_colbert-v2-hnp-msmarco/docid")
OUTPUT = Path("data/evaluation/runs/tenancy-dense-faiss.txt")

TOP_K = 5


def main():
    searcher = FaissSearcher(INDEX, QUERY_ENCODER)

    with DOCIDS.open("r", encoding="utf-8") as f:
        docids = [line.strip() for line in f if line.strip()]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with QUESTIONS.open("r", encoding="utf-8") as f, \
         OUTPUT.open("w", encoding="utf-8") as out:

        for line in f:
            item = json.loads(line)
            query_id = item["query_id"]
            query = item["query"]

            hits = searcher.search(query, TOP_K)

            for rank, hit in enumerate(hits, start=1):
                doc_id = hit.docid

                out.write(
                    f"{query_id} Q0 {doc_id} {rank} "
                    f"{hit.score:.6f} tenancy-dense-faiss\n"
                )

            print(f"{query_id}: {len(hits)} results")


if __name__ == "__main__":
    main()

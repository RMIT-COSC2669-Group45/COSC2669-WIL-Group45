from pyserini.search.lucene import LuceneSearcher
import sys

INDEX = "target/indexes/tenancy-bm25"

def search(query, k=5):
    searcher = LuceneSearcher(INDEX)
    hits = searcher.search(query, k)

    print(f"\nQuery: {query}")
    print("-" * 80)

    for rank, hit in enumerate(hits, start=1):
        doc = searcher.doc(hit.docid)

        print(f"\nRank {rank}")
        print(f"Document ID: {hit.docid}")
        print(f"Score: {hit.score:.4f}")
        print(f"Content:\n{doc.raw()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/retrieval/search-tenancy.py "your question"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search(query)

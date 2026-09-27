#!/bin/bash

CORPUS="data/tenancy_collection"
INDEX="target/indexes/tenancy-bm25"

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$CORPUS" \
  --language en \
  --index "$INDEX" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions \
  --storeDocvectors \
  --storeRaw

import streamlit as st

from pyserini.search import LuceneSearcher, FaissSearcher

from src.rag.chatbot import (
    BM25_INDEX,
    DENSE_INDEX,
    QUERY_ENCODER,
    load_documents,
    hybrid_search,
    build_prompt,
    generate_answer,
)


st.set_page_config(
    page_title="Victorian Tenancy Assistant",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_rag():
    """Load the existing E7.2 RAG components once."""

    documents = load_documents()

    bm25_searcher = LuceneSearcher(BM25_INDEX)

    dense_searcher = FaissSearcher(
        DENSE_INDEX,
        QUERY_ENCODER,
    )

    return documents, bm25_searcher, dense_searcher


def answer_question(question):
    """Run one question through the existing E7.2 RAG pipeline."""

    documents, bm25_searcher, dense_searcher = load_rag()

    retrieved = hybrid_search(
        bm25_searcher,
        dense_searcher,
        question,
        documents,
    )

    prompt = build_prompt(
        question,
        retrieved,
        documents,
    )

    answer = generate_answer(prompt)

    return answer, retrieved, documents


# ---------------------------------------------------------
# Interface
# ---------------------------------------------------------

st.title("🏠 Victorian Tenancy Assistant")

st.write(
    "Ask a question about Victorian residential tenancy rules "
    "using the project's Consumer Affairs Victoria knowledge base."
)

st.info(
    "This prototype provides information from the project's "
    "retrieved tenancy sources. It is not legal advice."
)


question = st.text_area(
    "Ask your tenancy question:",
    placeholder="For example: How much bond can a rental provider ask for?",
    height=100,
)


if st.button("Ask Question", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching the tenancy knowledge base..."):

            try:
                answer, retrieved, documents = answer_question(
                    question.strip()
                )

                st.subheader("Answer")

                st.write(answer)

                st.subheader("Sources")

                if retrieved:

                    for item in retrieved:

                        doc_id = item["id"]

                        with st.expander(doc_id):

                            st.write(
                                documents.get(
                                    doc_id,
                                    "Source text unavailable.",
                                )
                            )

                            if "score" in item:
                                st.caption(
                                    f"Hybrid score: {item['score']:.3f}"
                                )

                            if "topic_score" in item:
                                st.caption(
                                    f"Topic score: {item['topic_score']:.3f}"
                                )

                            if "evidence_score" in item:
                                st.caption(
                                    f"Evidence score: {item['evidence_score']:.3f}"
                                )

                else:
                    st.warning(
                        "No supporting sources were retrieved."
                    )

            except Exception as e:

                st.error(
                    "The RAG system could not generate an answer."
                )

                st.exception(e)
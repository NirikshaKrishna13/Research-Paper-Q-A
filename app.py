import os

# ------------------------------------------------------------
# Reduce CPU memory pressure on Windows
# ------------------------------------------------------------
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import tempfile
import re

import streamlit as st
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)

# Keep local CPU inference lightweight and prevent crashes on Streamlit reruns.
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 Research Paper Q&A")

st.write(
    "Upload a research paper and ask questions using "
    "local AI models. No OpenAI API or external LLM API is used."
)

st.info(
    "🔒 Local AI mode: PDF processing, embeddings, retrieval, "
    "question answering, summary, and key insights run on this computer."
)


# ============================================================
# LOCAL GENERATIVE MODEL — SUMMARY / INSIGHTS
# ============================================================

@st.cache_resource
def get_llm():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    model.eval()

    def generate_text(prompt, max_tokens=180):

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=2,
                no_repeat_ngram_size=3,
                do_sample=False,
                early_stopping=True,
            )

        return tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

    return generate_text


# ============================================================
# LOCAL EXTRACTIVE QA MODEL
# ============================================================

@st.cache_resource
def get_qa_model():

    # Small local QA model: about 81.5M parameters.
    model_name = "deepset/tinyroberta-squad2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name
    )

    model.eval()

    return tokenizer, model


# ============================================================
# LOCAL EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 2
        },
    )


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name

    try:

        # --------------------------------------------------------
        # 1. Load PDF
        # --------------------------------------------------------

        loader = PyPDFLoader(
            temp_path
        )

        documents = loader.load()

        if not documents:
            raise ValueError(
                "No readable text was found in the PDF."
            )

        # --------------------------------------------------------
        # 1A. Clean PDF text
        # --------------------------------------------------------
        # Some PDFs contain unusual/non-string page content.
        # SentenceTransformer tokenizers require plain strings.
        cleaned_documents = []

        for doc in documents:
            page_text = doc.page_content

            if page_text is None:
                continue

            if not isinstance(page_text, str):
                page_text = str(page_text)

            page_text = page_text.replace("\x00", " ").strip()

            if page_text:
                doc.page_content = page_text
                cleaned_documents.append(doc)

        documents = cleaned_documents

        if not documents:
            raise ValueError(
                "The PDF was loaded, but no readable text was found."
            )

        # --------------------------------------------------------
        # 2. Split into chunks
        # --------------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
        )

        chunks = text_splitter.split_documents(
            documents
        )

        # Make absolutely sure every embedding input is a string.
        cleaned_chunks = []

        for chunk in chunks:
            if chunk.page_content is None:
                continue

            if not isinstance(chunk.page_content, str):
                chunk.page_content = str(chunk.page_content)

            chunk.page_content = (
                chunk.page_content
                .replace("\x00", " ")
                .strip()
            )

            if chunk.page_content:
                cleaned_chunks.append(chunk)

        chunks = cleaned_chunks

        if not chunks:
            raise ValueError(
                "Could not create readable text chunks from the PDF."
            )

        # --------------------------------------------------------
        # 3. Create local embeddings
        # --------------------------------------------------------

        embeddings = get_embeddings()

        # --------------------------------------------------------
        # 4. Create FAISS vector database
        # --------------------------------------------------------

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        return documents, chunks, vectorstore

    finally:

        try:
            os.remove(temp_path)

        except Exception:
            pass


# ============================================================
# EXTRACT ANSWER SPAN DIRECTLY FROM CONTEXT
# ============================================================

def extract_answer(
    question,
    context,
    tokenizer,
    model
):
    """
    Direct local extractive QA.
    No Transformers pipeline is used.
    """

    question = str(question).strip()
    context = str(context).strip()

    if not question:
        return "", 0.0

    if not context:
        return "", 0.0

    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
        stride=64,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    answers = []

    number_of_features = encoded["input_ids"].shape[0]

    for feature_index in range(number_of_features):

        input_ids = encoded["input_ids"][
            feature_index:feature_index + 1
        ]

        attention_mask = encoded["attention_mask"][
            feature_index:feature_index + 1
        ]

        offset_mapping = encoded["offset_mapping"][
            feature_index
        ].tolist()

        sequence_ids = encoded.sequence_ids(
            feature_index
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():

            outputs = model(
                **model_inputs
            )

        start_probs = torch.softmax(
            outputs.start_logits[0],
            dim=0
        )

        end_probs = torch.softmax(
            outputs.end_logits[0],
            dim=0
        )

        context_positions = [
            i
            for i, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1
        ]

        if not context_positions:
            continue

        top_start_indices = torch.topk(
            start_probs[context_positions],
            k=min(8, len(context_positions))
        ).indices.tolist()

        top_end_indices = torch.topk(
            end_probs[context_positions],
            k=min(8, len(context_positions))
        ).indices.tolist()

        start_positions = [
            context_positions[index]
            for index in top_start_indices
        ]

        end_positions = [
            context_positions[index]
            for index in top_end_indices
        ]

        for start_position in start_positions:

            for end_position in end_positions:

                if end_position < start_position:
                    continue

                # Avoid absurdly long answer spans.
                if end_position - start_position > 30:
                    continue

                start_char = offset_mapping[
                    start_position
                ][0]

                end_char = offset_mapping[
                    end_position
                ][1]

                if end_char <= start_char:
                    continue

                answer = context[
                    start_char:end_char
                ].strip()

                if not answer:
                    continue

                score = (
                    start_probs[start_position].item()
                    *
                    end_probs[end_position].item()
                )

                answers.append(
                    {
                        "answer": answer,
                        "score": score,
                    }
                )

    if not answers:

        return "", 0.0

    # Sort by model confidence.
    answers.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    best = answers[0]

    return (
        best["answer"],
        best["score"]
    )


# ============================================================
# ANSWER QUESTION — RAG
# ============================================================

# ============================================================
# HELPER: EXTRACT ABSTRACT DIRECTLY FROM PAPER
# ============================================================

def extract_abstract_text(documents):
    """Extract abstract text directly from the first pages if available."""
    if not documents:
        return ""
    candidates = documents[:2]
    for doc in candidates:
        text = doc.page_content or ""
        match = re.search(
            r'\babstract\b[\s:\.\-—]*(.+?)(?=\b(?:keywords|index terms|1\.?\s+introduction|introduction)\b|\n\s*\n\s*[0-9]\.|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            clean = " ".join(match.group(1).strip().split())
            if len(clean) > 60:
                return clean[:1600]
    return ""


# ============================================================
# HELPER: EXTRACT KEY SENTENCES FROM CHUNKS
# ============================================================

def extract_key_sentences(query, docs, max_sentences=2):
    """Extract the most relevant factual sentences from docs matching query keywords."""
    if not docs:
        return []
    keywords = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', query))
    candidates = []
    for doc in docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content.replace('\n', ' '))
        for s in sentences:
            s_clean = " ".join(s.strip().split())
            if 35 <= len(s_clean) <= 300:
                words = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', s_clean))
                overlap = len(keywords.intersection(words))
                if overlap > 0:
                    candidates.append((overlap, s_clean))
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    result = []
    for _score, s in candidates:
        norm = s[:45].lower()
        if norm not in seen:
            seen.add(norm)
            result.append(s)
            if len(result) >= max_sentences:
                break
    return result


# ============================================================
# ANSWER QUESTION — HYBRID RAG (EXTRACTIVE + GENERATIVE)
# ============================================================

def answer_question(
    question,
    vectorstore
):
    q_lower = question.lower().strip()

    # 1. Retrieve relevant chunks from FAISS
    retrieved = vectorstore.similarity_search_with_score(
        question,
        k=5
    )

    if not retrieved:
        return (
            "No matching sections were found in the uploaded paper.",
            []
        )

    docs = [doc for doc, _score in retrieved]
    llm = get_llm()
    qa_tokenizer, qa_model = get_qa_model()

    # 2. Check for broad overview / summary / insight questions
    is_overview_query = any(k in q_lower for k in [
        "summar", "overview", "insight", "what is this paper about",
        "main idea", "explain the paper", "tell me about", "takeaway",
        "contribution", "conclusion"
    ])

    if is_overview_query:
        top_context = " ".join([d.page_content.strip() for d in docs[:3]])
        top_snippet = " ".join(top_context.split()[:180])
        gen_answer = llm(f"summarize: {top_snippet}", max_tokens=160)

        key_sentences = extract_key_sentences(question, docs[:3], max_sentences=3)
        parts = []
        if gen_answer and len(gen_answer) > 20:
            parts.append(gen_answer)
        if key_sentences:
            parts.extend([f"• {s}" for s in key_sentences])

        if parts:
            return "\n\n".join(parts), docs

    # 3. For specific/factoid questions, attempt extractive QA
    candidate_answers = []
    for rank, (doc, retrieval_score) in enumerate(
        retrieved,
        start=1
    ):
        context = doc.page_content.strip()
        if not context:
            continue

        try:
            answer, confidence = extract_answer(
                question,
                context,
                qa_tokenizer,
                qa_model
            )

            if (
                answer
                and len(answer.strip()) >= 2
                and confidence >= 0.05
            ):
                candidate_answers.append(
                    {
                        "answer": answer.strip(),
                        "confidence": confidence,
                        "rank": rank,
                        "doc": doc,
                    }
                )
        except Exception:
            continue

    if candidate_answers:
        candidate_answers.sort(
            key=lambda item: (
                item["confidence"],
                -item["rank"]
            ),
            reverse=True
        )
        best_answer = candidate_answers[0]["answer"].strip(' "\'\n\t')
        if len(best_answer) >= 2 and best_answer.lower() not in {"the", "it", "this", "yes", "no"}:
            return best_answer, docs

    # 4. Generative fallback using local FLAN-T5
    top_context = " ".join([d.page_content.strip() for d in docs[:2]])
    top_snippet = " ".join(top_context.split()[:180])
    gen_answer = llm(f"question: {question} context: {top_snippet}", max_tokens=120)

    if gen_answer and len(gen_answer) > 5 and gen_answer.lower() not in {"none", "unanswerable", "no answer"}:
        return gen_answer, docs

    # 5. Fallback to key contextual sentences from top chunks
    fallback_sentences = extract_key_sentences(question, docs[:2], max_sentences=2)
    if fallback_sentences:
        return (
            "Key context identified from the paper:\n\n"
            + "\n".join([f"• \"{s}\"" for s in fallback_sentences]),
            docs
        )

    return (
        "The answer could not be confidently identified in the retrieved sections. Please review the retrieved chunks below.",
        docs
    )


# ============================================================
# LOCAL TEXT GENERATION HELPER
# ============================================================

def ask_llm(prompt, max_tokens=180):

    llm = get_llm()

    return llm(
        prompt,
        max_tokens=max_tokens
    )


# ============================================================
# GENERATE SUMMARY (STRUCTURED MULTI-SECTION)
# ============================================================

def generate_summary(vectorstore):

    documents = st.session_state.get("documents", [])
    llm = get_llm()

    sections = []

    # 1. Executive Abstract / Overview
    abstract = extract_abstract_text(documents)
    if abstract:
        abstract_snippet = " ".join(abstract.split()[:180])
        ai_summary = llm(f"summarize: {abstract_snippet}", max_tokens=140)

        overview_text = ""
        if ai_summary and len(ai_summary) > 20:
            overview_text += f"**AI Summary:** {ai_summary}\n\n"
        overview_text += f"**Original Abstract:**\n> {abstract}"
        sections.append(("📋 Executive Overview & Abstract", overview_text))
    else:
        top_chunks = vectorstore.similarity_search("abstract introduction overview background", k=2)
        combined = " ".join([d.page_content for d in top_chunks])
        snippet = " ".join(combined.split()[:180])
        ai_summary = llm(f"summarize: {snippet}", max_tokens=140)
        key_sentences = extract_key_sentences("objective background overview introduction", top_chunks, max_sentences=2)
        overview_parts = []
        if ai_summary and len(ai_summary) > 20:
            overview_parts.append(ai_summary)
        if key_sentences:
            overview_parts.extend([f"• {s}" for s in key_sentences])
        sections.append(("📋 Executive Overview", "\n\n".join(overview_parts) or "Overview extracted from document context."))

    # 2. Research Problem & Objective
    prob_chunks = vectorstore.similarity_search("research problem motivation objective goal challenge", k=2)
    prob_combined = " ".join([d.page_content for d in prob_chunks])
    prob_snippet = " ".join(prob_combined.split()[:180])
    prob_ai = llm(f"summarize: {prob_snippet}", max_tokens=120)
    prob_sentences = extract_key_sentences("problem objective goal address challenge overcome", prob_chunks, max_sentences=2)
    prob_parts = []
    if prob_ai and len(prob_ai) > 15:
        prob_parts.append(prob_ai)
    if prob_sentences:
        prob_parts.extend([f"• {s}" for s in prob_sentences])
    sections.append(("🎯 Research Problem & Objectives", "\n\n".join(prob_parts) or "Problem details identified in paper."))

    # 3. Methodology & Approach
    method_chunks = vectorstore.similarity_search("proposed method methodology architecture framework algorithm", k=2)
    method_combined = " ".join([d.page_content for d in method_chunks])
    method_snippet = " ".join(method_combined.split()[:180])
    method_ai = llm(f"summarize: {method_snippet}", max_tokens=120)
    method_sentences = extract_key_sentences("method approach architecture pipeline proposed model algorithm", method_chunks, max_sentences=2)
    method_parts = []
    if method_ai and len(method_ai) > 15:
        method_parts.append(method_ai)
    if method_sentences:
        method_parts.extend([f"• {s}" for s in method_sentences])
    sections.append(("⚙️ Methodology & Architecture", "\n\n".join(method_parts) or "Methodology details identified in paper."))

    # 4. Key Findings & Results
    results_chunks = vectorstore.similarity_search("experimental results evaluation performance findings accuracy", k=2)
    results_combined = " ".join([d.page_content for d in results_chunks])
    results_snippet = " ".join(results_combined.split()[:180])
    results_ai = llm(f"summarize: {results_snippet}", max_tokens=120)
    results_sentences = extract_key_sentences("results performance accuracy evaluation shows outperforms", results_chunks, max_sentences=2)
    results_parts = []
    if results_ai and len(results_ai) > 15:
        results_parts.append(results_ai)
    if results_sentences:
        results_parts.extend([f"• {s}" for s in results_sentences])
    sections.append(("📊 Important Findings & Results", "\n\n".join(results_parts) or "Results details identified in paper."))

    # 5. Conclusion & Takeaways
    concl_chunks = vectorstore.similarity_search("conclusion concluding remarks future work", k=2)
    concl_combined = " ".join([d.page_content for d in concl_chunks])
    concl_snippet = " ".join(concl_combined.split()[:180])
    concl_ai = llm(f"summarize: {concl_snippet}", max_tokens=120)
    concl_sentences = extract_key_sentences("conclusion concluded show demonstrated future", concl_chunks, max_sentences=2)
    concl_parts = []
    if concl_ai and len(concl_ai) > 15:
        concl_parts.append(concl_ai)
    if concl_sentences:
        concl_parts.extend([f"• {s}" for s in concl_sentences])
    sections.append(("🏁 Conclusion", "\n\n".join(concl_parts) or "Conclusion details identified in paper."))

    output_lines = []
    for title, content in sections:
        output_lines.append(f"#### {title}\n{content.strip()}\n")

    return "\n---\n".join(output_lines)


# ============================================================
# GENERATE KEY INSIGHTS (STRUCTURED HIGHLIGHTS)
# ============================================================

def generate_key_insights(vectorstore):

    documents = st.session_state.get("documents", [])
    llm = get_llm()

    insights = []

    # 1. Primary Contributions
    contrib_chunks = vectorstore.similarity_search("main contribution we propose our work introduces", k=2)
    contrib_combined = " ".join([d.page_content for d in contrib_chunks])
    contrib_snippet = " ".join(contrib_combined.split()[:180])
    contrib_ai = llm(f"summarize: {contrib_snippet}", max_tokens=120)
    contrib_sentences = extract_key_sentences("contribution propose introduce novel present", contrib_chunks, max_sentences=3)
    c_parts = []
    if contrib_ai and len(contrib_ai) > 15:
        c_parts.append(f"**Key Takeaway:** {contrib_ai}")
    if contrib_sentences:
        c_parts.extend([f"• {s}" for s in contrib_sentences])
    insights.append(("🌟 Primary Contributions", "\n\n".join(c_parts) or "Core contributions extracted from context."))

    # 2. Technical Innovation & Approach
    inno_chunks = vectorstore.similarity_search("novel architecture innovative technique algorithm implementation", k=2)
    inno_combined = " ".join([d.page_content for d in inno_chunks])
    inno_snippet = " ".join(inno_combined.split()[:180])
    inno_ai = llm(f"summarize: {inno_snippet}", max_tokens=120)
    inno_sentences = extract_key_sentences("algorithm framework model technique architecture novel", inno_chunks, max_sentences=3)
    i_parts = []
    if inno_ai and len(inno_ai) > 15:
        i_parts.append(f"**Key Takeaway:** {inno_ai}")
    if inno_sentences:
        i_parts.extend([f"• {s}" for s in inno_sentences])
    insights.append(("🔬 Technical Innovation & Approach", "\n\n".join(i_parts) or "Technical highlights extracted from context."))

    # 3. Quantitative Results & Performance
    perf_chunks = vectorstore.similarity_search("quantitative benchmark accuracy improvement performance baseline", k=2)
    perf_combined = " ".join([d.page_content for d in perf_chunks])
    perf_snippet = " ".join(perf_combined.split()[:180])
    perf_ai = llm(f"summarize: {perf_snippet}", max_tokens=120)
    perf_sentences = extract_key_sentences("percent improvement achieves outperforms score accuracy baseline", perf_chunks, max_sentences=3)
    p_parts = []
    if perf_ai and len(perf_ai) > 15:
        p_parts.append(f"**Key Takeaway:** {perf_ai}")
    if perf_sentences:
        p_parts.extend([f"• {s}" for s in perf_sentences])
    insights.append(("📈 Quantitative Results & Performance", "\n\n".join(p_parts) or "Performance metrics extracted from context."))

    # 4. Limitations & Future Directions
    limit_chunks = vectorstore.similarity_search("limitations drawbacks future work future directions challenges", k=2)
    limit_combined = " ".join([d.page_content for d in limit_chunks])
    limit_snippet = " ".join(limit_combined.split()[:180])
    limit_ai = llm(f"summarize: {limit_snippet}", max_tokens=120)
    limit_sentences = extract_key_sentences("limitation future work challenge constraint remain drawback", limit_chunks, max_sentences=3)
    l_parts = []
    if limit_ai and len(limit_ai) > 15:
        l_parts.append(f"**Key Takeaway:** {limit_ai}")
    if limit_sentences:
        l_parts.extend([f"• {s}" for s in limit_sentences])
    insights.append(("⚠️ Limitations & Future Directions", "\n\n".join(l_parts) or "Directions extracted from discussion context."))

    output_lines = []
    for title, content in insights:
        output_lines.append(f"#### {title}\n{content.strip()}\n")

    return "\n---\n".join(output_lines)


# ============================================================
# SIDEBAR — UPLOAD
# ============================================================

with st.sidebar:

    st.header(
        "📄 Upload Research Paper"
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    st.markdown("---")

    st.write("### 🛠️ Local Technology")

    st.write(
        """
        • Python
        • Streamlit
        • LangChain
        • FAISS
        • all-MiniLM-L6-v2
        • TinyRoBERTa-SQuAD2
        • FLAN-T5-small
        """
    )

    st.markdown("---")

    st.caption(
        "No OpenAI API. No Hugging Face API. "
        "No external LLM API."
    )


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    file_identifier = (
        uploaded_file.name,
        uploaded_file.size
    )

    if (
        "file_identifier" not in st.session_state
        or st.session_state.file_identifier
        != file_identifier
    ):

        st.session_state.file_identifier = (
            file_identifier
        )

        st.session_state.documents = None
        st.session_state.chunks = None
        st.session_state.vectorstore = None
        st.session_state.summary_result = None
        st.session_state.insights_result = None

        with st.spinner(
            "📖 Processing PDF: extracting text, creating chunks, "
            "embeddings, and FAISS index..."
        ):

            try:

                documents, chunks, vectorstore = (
                    process_pdf(uploaded_file)
                )

                st.session_state.documents = documents
                st.session_state.chunks = chunks
                st.session_state.vectorstore = vectorstore

                st.success(
                    f"✅ PDF processed: "
                    f"{len(documents)} pages | "
                    f"{len(chunks)} chunks"
                )

            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF:\n\n{e}"
                )


# ============================================================
# CHECK DOCUMENT
# ============================================================

if (
    "vectorstore" not in st.session_state
    or st.session_state.vectorstore is None
):

    st.markdown(
        "## 👈 Upload a research paper"
    )

    st.write(
        """
        After upload, the application will:

        1. Extract text from the PDF
        2. Split the paper into chunks
        3. Create local embeddings
        4. Store vectors in FAISS
        5. Retrieve relevant chunks
        6. Extract the answer using local QA
        """
    )

    st.stop()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💬 Ask Questions",
        "🧩 Chunks",
        "📝 Summary",
        "💡 Key Insights",
        "📄 Document Text",
    ]
)


# ============================================================
# TAB 1 — ASK QUESTIONS
# ============================================================

with tab1:

    st.header(
        "💬 Ask Questions"
    )

    st.write(
        "Ask a question about the uploaded research paper."
    )

    question = st.text_input(
        "Enter your question:",
        placeholder=(
            "Example: What methodology was used "
            "in this research?"
        )
    )

    if st.button(
        "🔍 Get Exact Answer",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "🔎 Retrieving relevant chunks and extracting the answer..."
            ):

                try:

                    answer, relevant_docs = answer_question(
                        question,
                        st.session_state.vectorstore
                    )

                    st.subheader(
                        "💬 Answer"
                    )

                    st.success(
                        answer
                    )

                    if relevant_docs:

                        st.subheader(
                            "🔍 Retrieved Chunks"
                        )

                        st.write(
                            f"FAISS retrieved "
                            f"**{len(relevant_docs)} chunks**."
                        )

                        for i, doc in enumerate(
                            relevant_docs,
                            start=1
                        ):

                            page = doc.metadata.get(
                                "page",
                                "Unknown"
                            )

                            if isinstance(
                                page,
                                int
                            ):
                                page_display = page + 1
                            else:
                                page_display = page

                            with st.expander(
                                f"🧩 Retrieved Chunk {i} "
                                f"— Page {page_display}",
                                expanded=True
                            ):

                                st.markdown(
                                    f"### Retrieved Chunk {i}"
                                )

                                st.markdown(
                                    f"**📄 Page:** "
                                    f"{page_display}"
                                )

                                st.text_area(
                                    f"Chunk {i}",
                                    doc.page_content,
                                    height=350,
                                    key=(
                                        f"retrieved_"
                                        f"{i}"
                                    )
                                )

                except Exception as e:

                    st.error(
                        f"❌ Error generating answer:\n\n{e}"
                    )


# ============================================================
# TAB 2 — ALL CHUNKS
# ============================================================

with tab2:

    st.header(
        "🧩 Document Chunks"
    )

    chunks = st.session_state.chunks

    st.write(
        f"Total chunks created: "
        f"**{len(chunks)}**"
    )

    st.info(
        "The chunks below are the text units embedded "
        "and indexed in FAISS."
    )

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        page = chunk.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(
            page,
            int
        ):
            page_display = page + 1
        else:
            page_display = page

        with st.expander(
            f"📦 Chunk {i} — Page {page_display}",
            expanded=False
        ):

            st.text_area(
                f"Chunk {i} Text",
                chunk.page_content,
                height=300,
                key=f"all_chunk_{i}"
            )


# ============================================================
# TAB 3 — SUMMARY
# ============================================================

with tab3:

    st.header(
        "📝 Research Paper Summary"
    )

    if st.button(
        "Generate Summary",
        type="primary"
    ):

        with st.spinner(
            "🤖 Generating local summary..."
        ):

            try:

                st.session_state.summary_result = generate_summary(
                    st.session_state.vectorstore
                )

            except Exception as e:

                st.error(
                    f"❌ Error generating summary:\n\n{e}"
                )

    if st.session_state.get("summary_result"):
        st.markdown(
            st.session_state.summary_result
        )


# ============================================================
# TAB 4 — KEY INSIGHTS
# ============================================================

with tab4:

    st.header(
        "💡 Key Insights"
    )

    if st.button(
        "Extract Key Insights",
        type="primary"
    ):

        with st.spinner(
            "🤖 Extracting insights locally..."
        ):

            try:

                st.session_state.insights_result = generate_key_insights(
                    st.session_state.vectorstore
                )

            except Exception as e:

                st.error(
                    f"❌ Error extracting insights:\n\n{e}"
                )

    if st.session_state.get("insights_result"):
        st.markdown(
            st.session_state.insights_result
        )


# ============================================================
# TAB 5 — DOCUMENT TEXT
# ============================================================

with tab5:

    st.header(
        "📄 Extracted Document Text"
    )

    documents = st.session_state.documents

    for i, doc in enumerate(
        documents,
        start=1
    ):

        page_number = doc.metadata.get(
            "page",
            i - 1
        )

        if isinstance(
            page_number,
            int
        ):
            page_display = page_number + 1
        else:
            page_display = page_number

        with st.expander(
            f"Page {page_display}"
        ):

            st.write(
                doc.page_content
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Research Paper Q&A | "
    "Local RAG Application | "
    "No External LLM API"
)

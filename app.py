"""
Marine Engine Troubleshooting AI Agent
--------------------------------------
Beginner-friendly Streamlit application using:
- Streamlit
- LangChain
- FAISS
- Local Hugging Face embeddings
- Groq-hosted LLM

Place one or more authorized OEM PDFs in ./data/.
The PDF filename should contain the supported engine model name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

APP_TITLE = "Marine Engine Troubleshooting AI Agent"

DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstores")

# Groq's current model catalog/deprecation guidance should be checked
# before deployment. This is the recommended replacement for the
# deprecated Llama 3.3 70B model in the current Groq documentation.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ENGINE_MODELS = [
    "MTU 10V 2000 M94",
    "MTU 12V 2000 M94",
    "MTU 12V 2000 M96L",
    "MTU 16V 4000 M90",
    "MAN 12V 175D",
    "MAN 16V 175D",
]

# A few normalization aliases make matching PDF filenames easier.
ENGINE_ALIASES = {
    "MTU 10V 2000 M94": ["mtu 10v 2000 m94"],
    "MTU 12V 2000 M94": ["mtu 12v 2000 m94"],
    "MTU 12V 2000 M96L": ["mtu 12v 2000 m96l"],
    "MTU 16V 4000 M90": ["mtu 16v 4000 m90"],
    "MAN 12V 175D": ["man 12v 175d"],
    "MAN 16V 175D": ["man 16v 175d"],
}


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

def apply_css() -> None:
    """Apply simple industrial styling without external CSS files."""
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }
        .subtitle {
            color: #667085;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }
        .status-card {
            border: 1px solid #d0d5dd;
            border-radius: 12px;
            padding: 14px 16px;
            background: #f8fafc;
        }
        .warning-card {
            border-left: 5px solid #d97706;
            padding: 12px 15px;
            background: #fffbeb;
            border-radius: 6px;
        }
        .source-card {
            border: 1px solid #d0d5dd;
            border-radius: 10px;
            padding: 12px;
            margin: 8px 0;
            background: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize text for reliable filename matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_engine_pdfs(engine_model: str) -> List[Path]:
    """Find local PDFs whose names contain the selected engine model."""
    if not DATA_DIR.exists():
        return []

    wanted = normalize(engine_model)
    aliases = [normalize(x) for x in ENGINE_ALIASES.get(engine_model, [])]

    matches: List[Path] = []
    for pdf in DATA_DIR.glob("*.pdf"):
        filename = normalize(pdf.stem)
        if wanted in filename or any(alias in filename for alias in aliases):
            matches.append(pdf)

    return sorted(matches)


def source_label(doc: Document) -> str:
    """Create a human-readable source label."""
    source = Path(str(doc.metadata.get("source", "Unknown document"))).name
    page = doc.metadata.get("page")
    if isinstance(page, int):
        return f"{source} — PDF page {page + 1}"
    return source


# ---------------------------------------------------------------------
# Embeddings and FAISS
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading local embedding model...")
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load embeddings locally.

    This does not send the manual text to a third-party embedding API.
    The model is downloaded once and cached by Hugging Face.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_split_pdfs(pdf_paths: List[Path]) -> List[Document]:
    """Load PDFs and split them into retrieval-friendly chunks."""
    if not pdf_paths:
        return []

    all_pages: List[Document] = []

    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            for page in pages:
                page.metadata["engine_model"] = pdf_path.stem
                page.metadata["document_name"] = pdf_path.name

            all_pages.extend(pages)
        except Exception as exc:
            raise RuntimeError(f"Could not read {pdf_path.name}: {exc}") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    chunks = splitter.split_documents(all_pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def build_vectorstore(engine_model: str) -> Tuple[FAISS, int]:
    """Build a FAISS index for one engine from its local PDFs."""
    pdfs = find_engine_pdfs(engine_model)

    if not pdfs:
        raise FileNotFoundError(
            f"No PDF manual found for '{engine_model}'. "
            f"Add an authorized PDF to {DATA_DIR.resolve()} "
            f"with the engine model in its filename."
        )

    chunks = load_and_split_pdfs(pdfs)

    if not chunks:
        raise RuntimeError(
            f"The PDF(s) for {engine_model} produced no readable text."
        )

    vectorstore = FAISS.from_documents(chunks, get_embeddings())

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = VECTORSTORE_DIR / normalize(engine_model).replace(" ", "_")
    vectorstore.save_local(str(save_path))

    return vectorstore, len(chunks)


def vectorstore_path(engine_model: str) -> Path:
    return VECTORSTORE_DIR / normalize(engine_model).replace(" ", "_")


@st.cache_resource(show_spinner="Loading FAISS knowledge base...")
def get_vectorstore(engine_model: str) -> FAISS:
    """Load an existing FAISS index, building it if necessary."""
    path = vectorstore_path(engine_model)
    index_file = path / "index.faiss"

    if index_file.exists():
        return FAISS.load_local(
            str(path),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    vectorstore, _ = build_vectorstore(engine_model)
    return vectorstore


# ---------------------------------------------------------------------
# Groq / RAG generation
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an industrial marine-engine troubleshooting assistant.

Your job is to help a qualified marine engineer troubleshoot the selected
engine using ONLY the supplied OEM/manual context.

CRITICAL RULES:
1. Treat the retrieved manual context as the authoritative technical source.
2. Do not invent procedures, measurements, tolerances, pressure limits,
   temperatures, torque values, alarm meanings, part numbers, or safety rules.
3. If the manual context does not support an answer, explicitly say:
   "The supplied manual context is insufficient to verify this."
4. Do not claim that a fault is confirmed unless the supplied evidence
   supports that conclusion.
5. Clearly distinguish:
   - information directly supported by the manual,
   - reasonable diagnostic interpretation,
   - information that still needs verification.
6. Give a practical troubleshooting sequence, but do not create steps that
   are not supported by the supplied context.
7. Safety-critical actions must be presented as requiring qualified-engineer
   and OEM/procedure verification.
8. Cite the supplied source after important technical claims using the
   source labels included in the context.
9. If several causes are possible, rank them only when the supplied evidence
   gives a basis for doing so.
10. Do not use outside knowledge to fill gaps.

OUTPUT FORMAT:

## Assessment
Briefly restate the reported problem.

## Potential Root Causes
List the plausible causes supported by the retrieved manual context.

## Troubleshooting Sequence
Provide numbered steps. For each step include:
- Check/action
- What to observe or measure, if the manual states it
- Expected result/limit, only if explicitly supported
- Source

## Safety Precautions
Only state precautions supported by the retrieved context. Also remind the
engineer to follow the current approved OEM procedure and site safety rules.

## Evidence & Limitations
State what the manual evidence supports and what cannot be verified.
"""


def make_context(docs: List[Document]) -> str:
    """Convert retrieved documents into a clearly separated context block."""
    sections = []

    for number, doc in enumerate(docs, start=1):
        sections.append(
            f"[SOURCE {number}: {source_label(doc)}]\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n".join(sections)


def create_llm(api_key: str, model_name: str) -> ChatGroq:
    """Create a Groq chat model using the user-supplied key."""
    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.0,
        max_tokens=4000,
    )


def run_rag(
    api_key: str,
    engine_model: str,
    problem: str,
    model_name: str,
    top_k: int = 6,
) -> Tuple[str, List[Document]]:
    """Retrieve manual chunks and generate an evidence-grounded answer."""
    vectorstore = get_vectorstore(engine_model)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    docs = retriever.invoke(problem)

    if not docs:
        raise RuntimeError(
            "No relevant manual passages were retrieved for this problem."
        )

    context = make_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                """
Selected engine:
{engine_model}

Engineer-reported problem/symptoms:
{problem}

Retrieved OEM/manual context:
{context}

Provide the requested troubleshooting assessment.
""",
            ),
        ]
    )

    llm = create_llm(api_key, model_name)
    chain = prompt | llm

    response = chain.invoke(
        {
            "engine_model": engine_model,
            "problem": problem,
            "context": context,
        }
    )

    return response.content, docs


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

def render_sidebar() -> Tuple[str, str, str]:
    """Render controls and return selected values."""
    with st.sidebar:
        st.header("Configuration")

        engine_model = st.selectbox(
            "Engine model",
            ENGINE_MODELS,
            help="The selected model controls which local manual index is searched.",
        )

       def get_groq_api_key() -> str:
    """Get the Groq API key securely from Streamlit Secrets."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except KeyError:
        return ""
        )

        model_name = st.text_input(
            "Groq model",
            value=DEFAULT_GROQ_MODEL,
            help="You can change this when your Groq account uses another supported model.",
        )

        st.divider()
        st.caption("Knowledge source")
        st.write(
            "Local authorized OEM PDFs in `data/`. "
            "The app does not download manuals automatically."
        )

        st.caption("Safety")
        st.write(
            "Decision support only. Verify critical actions against the "
            "current OEM procedure and site safety requirements."
        )

    return engine_model, api_key.strip(), model_name.strip()


def render_sources(docs: List[Document]) -> None:
    """Display retrieved evidence sources."""
    st.subheader("Retrieved manual evidence")

    seen = set()
    for doc in docs:
        label = source_label(doc)
        if label in seen:
            continue
        seen.add(label)

        st.markdown(
            f"""
            <div class="source-card">
                <strong>{label}</strong><br>
                <span style="color:#667085;">
                Retrieved passage used by the AI response.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View retrieved passage"):
            st.write(doc.page_content)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_css()

    st.markdown(
        f'<div class="main-title">⚙️ {APP_TITLE}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">'
        "Evidence-grounded troubleshooting using authorized OEM manuals, "
        "FAISS retrieval and Groq AI."
        "</div>",
        unsafe_allow_html=True,
    )

    engine_model, api_key, model_name = render_sidebar()

    # Knowledge-base status
    pdfs = find_engine_pdfs(engine_model)
    index_exists = (vectorstore_path(engine_model) / "index.faiss").exists()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Selected engine", engine_model)
    with col2:
        st.metric("Local manuals", len(pdfs))
    with col3:
        st.metric("FAISS index", "Ready" if index_exists else "Build on first use")

    st.divider()

    st.subheader("Troubleshooting case")

    problem = st.text_area(
        "Describe the problem, alarm, symptoms, readings and operating conditions",
        height=180,
        placeholder=(
            "Example: Engine is showing an abnormal exhaust-temperature condition "
            "on one cylinder. Describe any alarm code, load, RPM, recent maintenance "
            "and measured values that are available."
        ),
    )

    left, right = st.columns([1, 4])
    with left:
        analyze = st.button(
            "🔎 Analyze problem",
            type="primary",
            use_container_width=True,
        )

    if not analyze:
        st.info(
            "Select an engine, enter the problem, provide your Groq API key, "
            "then click **Analyze problem**."
        )
        return

    if not api_key:
        st.error("Please enter your Groq API key.")
        return

    if not problem.strip():
        st.error("Please describe the engine problem or symptom.")
        return

    if not pdfs:
        st.error(
            f"No manual PDF was found for **{engine_model}**. "
            "Add the appropriate authorized PDF to the `data/` directory "
            "and try again."
        )
        st.stop()

    with st.spinner("Retrieving manual evidence and generating assessment..."):
        try:
            answer, docs = run_rag(
                api_key=api_key,
                engine_model=engine_model,
                problem=problem.strip(),
                model_name=model_name or DEFAULT_GROQ_MODEL,
            )

            st.success("Analysis completed.")

            st.subheader("AI troubleshooting assessment")
            st.markdown(answer)

            st.divider()
            render_sources(docs)

            st.divider()
            st.markdown(
                """
                <div class="warning-card">
                <strong>Engineering verification required.</strong><br>
                This application is decision support, not a replacement for
                the current OEM manual, approved maintenance procedure,
                safety rules, permits, isolation requirements, or qualified
                engineering judgment. Do not perform a critical action solely
                because the AI suggested it.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except FileNotFoundError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(f"Configuration error: {exc}")
        except Exception as exc:
            st.error(
                "The application could not complete the analysis. "
                "Check the API key, model name, local manual and FAISS index."
            )
            with st.expander("Technical error details"):
                st.exception(exc)


if __name__ == "__main__":
    main()

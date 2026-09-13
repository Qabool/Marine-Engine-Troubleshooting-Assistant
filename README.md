# Marine Engine Troubleshooting AI Agent

A beginner-friendly Streamlit application for AI-assisted troubleshooting of selected MTU and MAN marine engines.

The application follows the product requirements for an evidence-grounded troubleshooting assistant:

**Engine selection → local OEM manual → LangChain/FAISS retrieval → Groq LLM → structured troubleshooting response → source evidence**

> **Important:** This is an engineering decision-support prototype. It is not a replacement for the current OEM manual, approved maintenance procedure, safety rules, permits/isolation requirements, or qualified engineering judgment.

## 1. Supported engines

The first version supports:

- MTU 10V 2000 M94
- MTU 12V 2000 M94
- MTU 12V 2000 M96L
- MTU 16V 4000 M90
- MAN 12V 175D
- MAN 16V 175D

The code is designed so additional engines can be added later.

---

## 2. How the application works

```text
Authorized OEM PDF manuals
          |
          v
      PyPDFLoader
          |
          v
Text splitting / chunking
          |
          v
Local Hugging Face embeddings
          |
          v
       FAISS index
          |
          v
Retrieve relevant manual passages
          |
          v
      Groq LLM
          |
          v
Evidence-grounded troubleshooting
```

The LLM is instructed to use only the retrieved manual context for technical claims. If the retrieved evidence is insufficient, the application tells the model to say so instead of filling the gap from general knowledge.

---

## 3. Google Drive manuals

The project documentation specifies this Google Drive folder as the source location:

https://drive.google.com/drive/folders/1Lb-c_v5pyEdyWTizyrXJ_xwRjn87YhwA

The application intentionally does **not** automatically download files from Google Drive.

Instead:

1. Open the Google Drive folder.
2. Download the authorized maintenance manual PDFs you are permitted to use.
3. Create a folder called `data` beside `app.py`.
4. Put the PDFs inside `data`.
5. Make the engine model recognizable in each filename.

Recommended examples:

```text
data/
├── MTU_10V_2000_M94.pdf
├── MTU_12V_2000_M94.pdf
├── MTU_12V_2000_M96L.pdf
├── MTU_16V_4000_M90.pdf
├── MAN_12V_175D.pdf
└── MAN_16V_175D.pdf
```

If an engine has more than one authorized manual, you can keep multiple PDFs for that engine. For example:

```text
data/
├── MTU_16V_4000_M90_Maintenance.pdf
├── MTU_16V_4000_M90_Troubleshooting.pdf
└── MTU_16V_4000_M90_Service.pdf
```

The filename only needs to contain the supported engine model.

### Important

Use only manuals and technical documents that you are authorized to use. Do not upload confidential/proprietary material to an unauthorized public repository.

---

## 4. Prerequisites

Install:

- Python 3.10 or newer
- pip
- Internet access for the first installation/model download
- A Groq API key
- Authorized OEM manuals in `data/`

---

## 5. Project structure

After setup, your directory should look like:

```text
marine-engine-troubleshooting/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── MTU_10V_2000_M94.pdf
│   ├── MTU_12V_2000_M94.pdf
│   ├── MTU_12V_2000_M96L.pdf
│   ├── MTU_16V_4000_M90.pdf
│   ├── MAN_12V_175D.pdf
│   └── MAN_16V_175D.pdf
│
└── vectorstores/
    └── ...
```

`vectorstores/` is created automatically when the first engine is indexed.

---

## 6. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 7. Install dependencies

```bash
pip install -r requirements.txt
```

The first use of the local embedding model may download the Hugging Face model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

That embedding model runs locally. Your OEM PDF text is therefore not sent to a separate embedding API.

---

## 8. Get a Groq API key

Create/use your Groq API key through the Groq console:

https://console.groq.com/

The application asks for the key in a password-style Streamlit input.

Do not put your personal API key directly into `app.py`, `requirements.txt`, GitHub, or the README.

---

## 9. Groq model note

The original project concept suggested:

```text
llama-3.3-70b-versatile
```

However, Groq's current documentation lists `llama-3.3-70b-versatile` as deprecated as of August 16, 2026 and recommends newer replacements.

This implementation therefore defaults to:

```text
openai/gpt-oss-120b
```

The model is configurable from the Streamlit sidebar.

Check Groq's current model availability and your account's applicable limits before deployment:

https://console.groq.com/docs/models

If your Groq account supports a different current model, enter its model ID in the sidebar.

---

## 10. Run the application

From the project directory:

```bash
streamlit run app.py
```

Streamlit will display a local URL, normally similar to:

```text
http://localhost:8501
```

Open that URL in your browser.

---

## 11. First use

1. Select the engine model.
2. Enter your Groq API key.
3. Confirm the Groq model.
4. Enter the engine problem/symptoms.
5. Click **Analyze problem**.
6. Wait while the application:
   - loads the selected engine's FAISS index,
   - retrieves relevant manual passages,
   - sends only the selected-engine context to Groq,
   - generates the troubleshooting assessment.
7. Review the retrieved evidence shown below the answer.

### Example input

```text
Engine has an abnormal exhaust-temperature condition on one cylinder.
The engine is operating under load. Include the alarm code if available,
recent maintenance, RPM/load and any measured temperature or pressure.
```

Use actual measurements and alarm codes when available.

---

## 12. FAISS index behavior

The application builds one local FAISS index per engine model.

For example:

```text
vectorstores/
├── mtu_10v_2000_m94/
├── mtu_12v_2000_m94/
├── mtu_12v_2000_m96l/
├── mtu_16v_4000_m90/
├── man_12v_175d/
└── man_16v_175d/
```

On first use, the relevant PDF is loaded, split into chunks, embedded and stored locally.

Later runs reuse the saved FAISS index.

### When manuals change

If you replace or update a PDF, delete the corresponding engine folder under `vectorstores/`.

Example:

```text
vectorstores/mtu_16v_4000_m90/
```

Then run the application again. The index will be rebuilt from the current PDF.

For production, document revision/version tracking should be added before allowing multiple revisions of the same manual to coexist.

---

## 13. Retrieval design

The application uses:

- `PyPDFLoader` for PDF extraction
- `RecursiveCharacterTextSplitter` for chunking
- `HuggingFaceEmbeddings` for local semantic embeddings
- `FAISS` for local vector similarity search
- LangChain retriever for top-k evidence
- `ChatGroq` for generation

The default chunk settings are:

```text
chunk size: 1200 characters
overlap:    180 characters
top-k:      6 passages
```

These values are intentionally simple starting points. For a production deployment, evaluate them against a real troubleshooting test set.

---

## 14. Safety and hallucination controls

The system prompt contains explicit restrictions:

- Do not invent maintenance procedures.
- Do not invent limits or specifications.
- Do not invent part numbers.
- Do not present unsupported information as fact.
- Identify insufficient evidence.
- Distinguish manual evidence from AI interpretation.
- Require qualified-engineer/OEM verification for critical actions.

This is essential for an industrial troubleshooting application.

The application should still be treated as decision support. Always verify critical work against the current approved OEM documentation and applicable site safety procedures.

---

## 15. Important production considerations

This three-file version is intentionally compact and beginner-friendly.

Before production deployment, consider adding:

1. Google Drive API integration with controlled authentication.
2. Document revision/version management.
3. Hybrid keyword + vector retrieval.
4. A reranking model.
5. Fault-code indexing.
6. Structured diagnostic session memory.
7. User authentication and role-based access.
8. Audit logging.
9. Evaluation/regression test cases.
10. Retrieval/citation accuracy testing.
11. Automated document ingestion.
12. Monitoring and error logging.
13. Secure server-side API-key management instead of per-user keys.
14. More robust OCR for scanned manuals.
15. Page-level source linking.
16. Explicit document approval/status metadata.

Do not expose proprietary OEM manuals through a public GitHub repository.

---

## 16. Troubleshooting common problems

### "No manual PDF was found"

Check that:

- the PDF is inside `data/`;
- the filename contains the selected engine model;
- the file has a `.pdf` extension.

### "The PDF produced no readable text"

The PDF may be scanned/image-only. This version expects extractable PDF text. Add an OCR pipeline for scanned manuals before production use.

### FAISS loading error after updating packages

Delete the relevant engine's folder under `vectorstores/` and allow the application to rebuild it.

### Groq authentication error

Check:

- API key is correct;
- the selected model is available to your Groq account;
- your account/project has permission to use that model.

### Slow first run

The local embedding model must be downloaded and the selected PDF must be indexed. Subsequent runs are faster because the FAISS index is reused.

---

## 17. Engineering limitation

A fluent AI response is not evidence that a diagnosis is correct.

The correct operating principle is:

```text
OEM evidence
     ↓
Retrieval
     ↓
AI interpretation
     ↓
Engineer verification
     ↓
Maintenance decision
```

The application should therefore be evaluated primarily on:

- retrieval correctness,
- source/citation correctness,
- groundedness,
- absence of invented technical values,
- diagnostic usefulness,
- and safe handling of insufficient evidence.


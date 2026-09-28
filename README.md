# 📄 Chat with PDF — LangChain RAG

A local **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions about their content through an interactive Streamlit chat interface.

The application uses **LangChain**, **FAISS**, **Hugging Face embeddings**, and **Llama 3.2 through Ollama** to retrieve relevant information from the uploaded PDF and generate grounded answers.

---

## 🚀 Features

* 📄 Upload PDF documents
* 🔍 Extract text from PDF files
* ✂️ Split documents into manageable text chunks
* 🧠 Generate local embeddings using Hugging Face
* 🗂️ Store embeddings using FAISS
* 🔎 Retrieve relevant document chunks using similarity search
* 🤖 Generate answers using Llama 3.2 through Ollama
* 💬 Interactive conversational chat
* 🧵 Multi-turn conversation support
* 📚 Conversation history
* 📑 Display source PDF pages
* 📊 Display retrieval distance scores
* 📋 Display PDF information
* 🗑️ Clear chat history
* 📥 Download chat history
* 🎛️ Adjustable number of retrieved chunks
* 🛡️ Ground answers in the uploaded PDF context

---

## 🏗️ RAG Architecture

```text
                User
                  │
                  ▼
            Upload PDF
                  │
                  ▼
            PyPDFLoader
                  │
                  ▼
            Text Extraction
                  │
                  ▼
     RecursiveCharacterTextSplitter
                  │
                  ▼
            Text Chunks
                  │
                  ▼
      Hugging Face Embeddings
                  │
                  ▼
          FAISS Vector Store
                  │
                  │
        ┌─────────▼─────────┐
        │   User Question   │
        └─────────┬─────────┘
                  │
                  ▼
          Similarity Search
                  │
                  ▼
       Relevant PDF Chunks
                  │
                  ▼
          RAG Prompt + Context
                  │
                  ▼
       Llama 3.2 via Ollama
                  │
                  ▼
       Grounded Answer
                  │
                  ▼
       Source PDF Pages
```

---

## 🛠️ Technologies Used

| Technology                     | Purpose                   |
| ------------------------------ | ------------------------- |
| Python                         | Application development   |
| LangChain                      | RAG application framework |
| PyPDF                          | PDF document loading      |
| RecursiveCharacterTextSplitter | Document chunking         |
| Hugging Face                   | Local text embeddings     |
| `all-MiniLM-L6-v2`             | Embedding model           |
| FAISS                          | Vector similarity search  |
| Ollama                         | Local LLM runtime         |
| Llama 3.2                      | Answer generation         |
| Streamlit                      | Web application interface |

---

## 🧠 How the Application Works

The application follows a Retrieval-Augmented Generation workflow.

### 1. PDF Upload

The user uploads a PDF through the Streamlit interface.

### 2. PDF Text Extraction

The application loads the PDF using `PyPDFLoader` and extracts the document text.

### 3. Text Chunking

The extracted text is divided into smaller overlapping chunks using:

```python
RecursiveCharacterTextSplitter
```

The current configuration uses:

```text
Chunk size: 1000
Chunk overlap: 200
```

### 4. Embedding Generation

Each document chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings run locally, so the application does not require an OpenAI embedding API.

### 5. FAISS Vector Store

The generated embeddings are stored in a FAISS vector database.

FAISS allows the application to search for document chunks that are semantically related to the user's question.

### 6. Question Retrieval

When the user asks a question, the application performs similarity search and retrieves the most relevant chunks from the PDF.

The number of retrieved chunks can be adjusted through the Streamlit interface.

### 7. Context Construction

The retrieved PDF content is combined into the context supplied to the language model.

The application also includes relevant conversation history so that follow-up questions can be understood.

### 8. Answer Generation

The retrieved context is sent to:

```text
Llama 3.2
```

through:

```text
Ollama
```

The prompt instructs the model to answer using the uploaded PDF context rather than inventing information.

### 9. Source Pages

The application displays the PDF pages associated with the retrieved documents so users can inspect where the information came from.

---

## 💻 Application Interface

The application provides:

### PDF Information

Displays:

* Uploaded PDF filename
* Number of pages
* Number of text chunks
* Processing status

### Chat Interface

Users can ask questions about the uploaded document and continue the conversation with follow-up questions.

### Source Pages

Each answer can display the PDF pages used during retrieval.

### Retrieval Scores

The application displays the raw FAISS retrieval distance scores for the retrieved chunks.

> These are raw distance values and should not be interpreted as percentages or probability scores.

### Chat Controls

Users can:

* Clear the conversation
* Download the conversation as a text file

---

## 📁 Project Structure

```text
Chat_with_PDF/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   └── PDF files for local testing
│
└── venv/
    └── Local Python virtual environment
```

### Important

The following files/folders are intentionally excluded from GitHub:

```text
venv/
data/
.env
```

The local PDF and Python virtual environment should not be committed to the repository.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rayankhan-ai/chat-with-pdf-rag.git
```

Move into the project directory:

```bash
cd chat-with-pdf-rag
```

---

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\Activate
```

---

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 🤖 Ollama Setup

This project uses Ollama to run Llama 3.2 locally.

Install Ollama on your computer and download the required model:

```powershell
ollama pull llama3.2
```

Make sure Ollama is running before using the application.

The application uses:

```text
Model: llama3.2
Context size: 2048
```

The reduced context size helps the application operate more reliably on local hardware.

---

## ▶️ Run the Application

Start Streamlit:

```powershell
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## 💬 How to Use

### Step 1

Open the Streamlit application.

### Step 2

Upload a PDF using the file uploader.

### Step 3

Wait until the application displays:

```text
PDF processed successfully
```

### Step 4

Review the PDF information panel.

### Step 5

Ask a question about the uploaded PDF.

### Step 6

Continue asking follow-up questions.

For example:

```text
What is machine learning?
```

Then:

```text
Explain it in simple words.
```

The conversation history helps the application understand the follow-up question.

### Step 7

Open the source section to inspect the retrieved PDF pages.

### Step 8

Use **Clear Chat** to start a new conversation.

### Step 9

Use **Download Chat** to save the conversation.

---

## 🛡️ Grounded RAG Responses

The application is designed to reduce unsupported answers by instructing the language model to use the retrieved PDF context.

The prompt follows these principles:

```text
1. Use information from the uploaded PDF context.
2. Do not intentionally rely on outside knowledge.
3. Do not invent information.
4. If the information is unavailable in the retrieved PDF context,
   clearly state that it is not available.
5. Use conversation history to understand follow-up questions.
```

This makes the application suitable for asking questions about documents such as:

* Study materials
* Lecture notes
* Research papers
* Technical documentation
* Reports
* Books
* Manuals

---

## 📊 Retrieval Scores

FAISS returns similarity-search distance values for retrieved documents.

For example:

```text
Retrieved Document 1 → Distance: ...
Retrieved Document 2 → Distance: ...
```

These values are useful for inspecting the retrieval process.

They are **not percentages** and should not be presented as:

```text
80% relevant
90% relevant
```

without an additional calibrated scoring method.

---

## 🔒 Local AI Processing

One important characteristic of this project is that the core AI workflow runs locally.

### Embeddings

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

locally.

### Vector Database

FAISS runs locally.

### Language Model

Llama 3.2 runs locally through Ollama.

Therefore, the core RAG workflow does not require an OpenAI API key.

---

## 🔐 Security

Do not commit sensitive credentials to GitHub.

The repository includes:

```text
.env.example
```

for configuration documentation.

Actual environment files such as:

```text
.env
```

are excluded through `.gitignore`.

Never commit real API keys, passwords, tokens, or other credentials.

---

## 🧪 Development and Testing

Before committing changes, check Python syntax:

```powershell
python -m py_compile app.py
```

If no output is returned, the syntax check passed.

Then run:

```powershell
streamlit run app.py
```

and test:

* PDF upload
* PDF processing
* Question answering
* Follow-up questions
* Source pages
* Retrieval scores
* Clear Chat
* Download Chat

---

## 🎯 Learning Outcomes

This project demonstrates practical knowledge of:

* Retrieval-Augmented Generation
* LangChain
* Large Language Models
* Local LLM deployment
* Document loading
* Text preprocessing
* Text chunking
* Embeddings
* Vector databases
* Semantic similarity search
* Prompt engineering
* Conversational AI
* Source attribution
* Streamlit application development
* Ollama
* Hugging Face Sentence Transformers
* FAISS

---

## 🔮 Future Improvements

Possible future improvements include:

* 📚 Support for multiple PDFs simultaneously
* 💾 Persistent FAISS indexes
* 🔄 Improved document retrieval
* 🔀 Hybrid keyword + vector search
* 🧠 Retrieval reranking
* ⚡ Streaming LLM responses
* 📑 More detailed source citations
* 📊 RAG evaluation metrics
* 👤 User authentication
* 💬 Multiple conversation sessions
* 📥 Export conversations as PDF
* ☁️ Production deployment
* 🔐 Advanced application security

---

## 📌 Project Highlights

This project demonstrates the complete flow of a practical RAG application:

```text
PDF
 ↓
Document Loading
 ↓
Text Splitting
 ↓
Embeddings
 ↓
FAISS
 ↓
Similarity Retrieval
 ↓
Context Construction
 ↓
Llama 3.2
 ↓
Grounded Answer
 ↓
Source Pages
```

It combines traditional document retrieval with a local Large Language Model to create an interactive document-question-answering system.

---

## 👨‍💻 Author

**Rayan Ahmad**

Aspiring Agentic AI Engineer

GitHub:

https://github.com/rayankhan-ai

---

## ⭐ Project

If you find this project useful, consider giving the repository a star on GitHub.

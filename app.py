# ============================================================
# CHAT WITH ANY PDF - RAG APPLICATION
# ============================================================
#
# This application allows users to:
#
# 1. Upload any PDF
# 2. Extract text from the PDF
# 3. Split the text into chunks
# 4. Create local embeddings
# 5. Store embeddings in FAISS
# 6. Retrieve relevant PDF chunks
# 7. Ask questions using Llama 3.2
# 8. Maintain multi-turn conversation history
# 9. Display PDF source pages
# 10. Display retrieval scores
# 11. Clear chat history
# 12. Download chat history
#
# Technologies:
# - Python
# - Streamlit
# - LangChain
# - PyPDF
# - Hugging Face Embeddings
# - FAISS
# - Ollama
# - Llama 3.2
#
# ============================================================


# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

# hashlib is used to create a unique ID for each uploaded PDF.
import hashlib

# tempfile creates a temporary file for the uploaded PDF.
import tempfile

# Streamlit creates the web interface.
import streamlit as st

# PyPDFLoader extracts text from PDF files.
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter divides PDF text into chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hugging Face creates local text embeddings.
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS stores and searches the document embeddings.
from langchain_community.vectorstores import FAISS

# ChatOllama connects LangChain with the local Ollama model.
from langchain_ollama import ChatOllama


# ============================================================
# STEP 2: STREAMLIT PAGE CONFIGURATION
# ============================================================

# Configure the Streamlit browser page.
st.set_page_config(
    page_title="Chat with Any PDF",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# STEP 3: APPLICATION TITLE
# ============================================================

# Display the main application title.
st.title("📚 Chat with Any PDF")

# Display a short description.
st.write(
    "Upload a PDF and ask questions about it using "
    "a local Retrieval-Augmented Generation (RAG) system."
)


# ============================================================
# STEP 4: INITIALIZE SESSION STATE
# ============================================================

# Store all previous chat messages.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store the uploaded PDF name.
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Store the number of PDF pages.
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = 0

# Store the number of text chunks.
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = 0

# Store a unique ID for the current PDF.
if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None

# Store the FAISS vector database.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ============================================================
# STEP 5: LOAD HUGGING FACE EMBEDDING MODEL
# ============================================================

# Cache the embedding model so it is not loaded again
# every time Streamlit reruns the application.
@st.cache_resource
def load_embeddings():

    # Load the local Hugging Face embedding model.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Try to load the embedding model.
try:

    embeddings = load_embeddings()

# Handle embedding model errors.
except Exception as error:

    st.error(
        "❌ Unable to load the embedding model."
    )

    st.exception(error)

    st.stop()


# ============================================================
# STEP 6: LOAD LOCAL LLAMA 3.2 MODEL
# ============================================================

# Cache the Llama model so it does not reload
# every time Streamlit reruns.
@st.cache_resource
def load_llm():

    # Connect LangChain to the local Llama 3.2 model.
    #
    # num_ctx=2048 keeps the context size smaller
    # and helps avoid Ollama memory problems.
    return ChatOllama(
        model="llama3.2",
        num_ctx=2048
    )


# Try to connect to Ollama.
try:

    llm = load_llm()

# Handle Ollama connection errors.
except Exception as error:

    st.error(
        "❌ Unable to connect to Ollama/Llama 3.2."
    )

    st.info(
        "Make sure Ollama is installed and the "
        "llama3.2 model is available."
    )

    st.exception(error)

    st.stop()


# ============================================================
# STEP 7: SIDEBAR SETTINGS
# ============================================================

# Create the sidebar.
with st.sidebar:

    # Sidebar heading.
    st.header("⚙️ Settings")

    # Allow the user to choose how many PDF chunks
    # should be retrieved for each question.
    retrieval_k = st.slider(
        "📌 Retrieved PDF Chunks",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    # Explain the slider.
    st.caption(
        "Controls how many relevant PDF chunks "
        "are sent to the LLM."
    )

    # Add a separator.
    st.divider()

    # Technology heading.
    st.subheader("🛠️ Technology")

    # Display technologies used by the project.
    st.write("🐍 Python")
    st.write("🔗 LangChain")
    st.write("📄 PyPDF")
    st.write("🧠 Hugging Face Embeddings")
    st.write("🔎 FAISS")
    st.write("🦙 Llama 3.2")
    st.write("🖥️ Ollama")
    st.write("🌐 Streamlit")


# ============================================================
# STEP 8: PDF UPLOAD
# ============================================================

# Create a PDF upload widget.
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# ============================================================
# STEP 9: PROCESS UPLOADED PDF
# ============================================================

# Continue only when the user uploads a PDF.
if uploaded_file is not None:

    # Read the uploaded PDF as bytes.
    pdf_bytes = uploaded_file.getvalue()

    # Check whether the uploaded file is empty.
    if len(pdf_bytes) == 0:

        st.error(
            "❌ The uploaded PDF is empty."
        )

        st.stop()

    # Create a unique ID based on the PDF content.
    #
    # If the same PDF is uploaded again,
    # the application will recognize it.
    current_pdf_id = hashlib.md5(
        pdf_bytes
    ).hexdigest()


    # ========================================================
    # STEP 10: CHECK WHETHER THIS IS A NEW PDF
    # ========================================================

    # Process the PDF only when it is different
    # from the previously uploaded PDF.
    if st.session_state.pdf_id != current_pdf_id:

        # Display processing information.
        st.info(
            f"📄 Processing: {uploaded_file.name}"
        )


        # ====================================================
        # STEP 11: PROCESS PDF SAFELY
        # ====================================================

        try:

            # Create a temporary PDF file.
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                # Write uploaded PDF bytes to the temporary file.
                temporary_file.write(pdf_bytes)

                # Store the temporary file path.
                pdf_path = temporary_file.name


            # =================================================
            # STEP 12: LOAD PDF
            # =================================================

            # Display a loading message.
            with st.spinner(
                "📄 Reading PDF..."
            ):

                # Create the PDF loader.
                loader = PyPDFLoader(
                    pdf_path
                )

                # Load all PDF pages.
                documents = loader.load()


            # Check whether pages were loaded.
            if not documents:

                st.error(
                    "❌ No readable pages were found "
                    "in this PDF."
                )

                st.stop()


            # Store the total number of PDF pages.
            st.session_state.pdf_pages = len(
                documents
            )


            # =================================================
            # STEP 13: SPLIT PDF INTO CHUNKS
            # =================================================

            # Display a loading message.
            with st.spinner(
                "🧩 Splitting PDF into text chunks..."
            ):

                # Create the text splitter.
                text_splitter = RecursiveCharacterTextSplitter(

                    # Maximum size of each chunk.
                    chunk_size=1000,

                    # Number of overlapping characters.
                    chunk_overlap=200
                )

                # Split the PDF into chunks.
                chunks = text_splitter.split_documents(
                    documents
                )


            # Check whether readable text was found.
            if not chunks:

                st.error(
                    "❌ No readable text was found "
                    "in the uploaded PDF."
                )

                st.info(
                    "This application currently works "
                    "with text-based PDFs. Scanned image-only "
                    "PDFs may require OCR."
                )

                st.stop()


            # Store the number of chunks.
            st.session_state.pdf_chunks = len(
                chunks
            )


            # =================================================
            # STEP 14: CREATE FAISS VECTOR DATABASE
            # =================================================

            # Display a loading message.
            with st.spinner(
                "🧠 Creating PDF knowledge base..."
            ):

                # Convert chunks into embeddings
                # and store them in FAISS.
                vector_store = FAISS.from_documents(
                    chunks,
                    embeddings
                )


            # Save the vector database in session state.
            st.session_state.vector_store = (
                vector_store
            )

            # Save the PDF name.
            st.session_state.pdf_name = (
                uploaded_file.name
            )

            # Save the PDF ID.
            st.session_state.pdf_id = (
                current_pdf_id
            )

            # Clear previous conversation because
            # a new PDF has been uploaded.
            st.session_state.messages = []


            # Tell the user that processing succeeded.
            st.success(
                "✅ PDF processed successfully!"
            )


        # ====================================================
        # STEP 15: HANDLE PDF PROCESSING ERRORS
        # ====================================================

        # This except MUST align with the try above.
        except Exception as error:

            # Display a user-friendly error.
            st.error(
                "❌ An error occurred while processing the PDF."
            )

            # Display the actual Python error.
            st.exception(error)

            # Stop the application safely.
            st.stop()


# ============================================================
# STEP 16: CHECK WHETHER PDF IS READY
# ============================================================

# Continue only when the FAISS vector store exists.
if st.session_state.vector_store is not None:

    # Add a horizontal divider.
    st.divider()

    # Display the current PDF name.
    st.subheader(
        f"💬 Chat with: {st.session_state.pdf_name}"
    )


    # ========================================================
    # STEP 17: CLEAR CHAT BUTTON
    # ========================================================

    # Create the Clear Chat button.
    if st.button(
        "🗑️ Clear Chat"
    ):

        # Delete all stored chat messages.
        st.session_state.messages = []

        # Refresh the Streamlit application.
        st.rerun()


    # ========================================================
    # STEP 18: DOWNLOAD CHAT
    # ========================================================

    # Start with an empty text variable.
    chat_text = ""

    # Loop through all chat messages.
    for message in st.session_state.messages:

        # Convert role to uppercase.
        role = message["role"].upper()

        # Get the message content.
        content = message["content"]

        # Add the message to the downloadable text.
        chat_text += (
            f"{role}:\n"
            f"{content}\n\n"
        )


    # Create the Download Chat button.
    st.download_button(
        label="📥 Download Chat",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )


    # ========================================================
    # STEP 19: PDF INFORMATION PANEL
    # ========================================================

    # Create an expandable PDF information panel.
    with st.expander(
        "📄 PDF Information",
        expanded=True
    ):

        # Display PDF filename.
        st.write(
            f"**File:** "
            f"{st.session_state.pdf_name}"
        )

        # Display number of pages.
        st.write(
            f"**Pages:** "
            f"{st.session_state.pdf_pages}"
        )

        # Display number of chunks.
        st.write(
            f"**Text Chunks:** "
            f"{st.session_state.pdf_chunks}"
        )

        # Display PDF status.
        st.write(
            "**Status:** "
            "✅ PDF Ready for Questions"
        )


    # ========================================================
    # STEP 20: DISPLAY PREVIOUS CHAT HISTORY
    # ========================================================

    # Display all previous messages.
    for message in st.session_state.messages:

        # Create a chat message container.
        with st.chat_message(
            message["role"]
        ):

            # Display the message content.
            st.write(
                message["content"]
            )


    # ========================================================
    # STEP 21: CHAT INPUT
    # ========================================================

    # Create the question input box.
    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    # ========================================================
    # STEP 22: PROCESS USER QUESTION
    # ========================================================

    # Continue only when the user enters a question.
    if question:

        # ----------------------------------------------------
        # Display user's question
        # ----------------------------------------------------

        with st.chat_message("user"):

            # Display the question.
            st.write(
                question
            )


        # Save the user's question to chat history.
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ====================================================
        # STEP 23: RETRIEVE RELEVANT PDF CHUNKS
        # ====================================================

        # Get the FAISS vector store.
        vector_store = (
            st.session_state.vector_store
        )


        # Search for relevant PDF chunks.
        #
        # similarity_search_with_score returns:
        # document + raw FAISS distance score.
        results_with_scores = (
            vector_store.similarity_search_with_score(
                question,
                k=retrieval_k
            )
        )


        # Create an empty list for documents.
        relevant_documents = []

        # Create an empty list for scores.
        similarity_scores = []


        # Process every retrieved result.
        for document, score in results_with_scores:

            # Save the document.
            relevant_documents.append(
                document
            )

            # Save the raw FAISS score.
            similarity_scores.append(
                score
            )


        # ====================================================
        # STEP 24: CREATE PDF CONTEXT
        # ====================================================

        # Start with an empty context.
        context = ""

        # Create a list for source page numbers.
        source_pages = []


        # Process each retrieved PDF document.
        for document in relevant_documents:

            # Add PDF text to the context.
            context += (
                document.page_content
            )

            # Add spacing between chunks.
            context += "\n\n"


            # Get the original PDF page number.
            page_number = (
                document.metadata.get("page")
            )


            # Check whether a page number exists.
            if page_number is not None:

                # PyPDF uses zero-based page numbers.
                #
                # Add 1 so the user sees normal
                # human-readable page numbers.
                page_number = (
                    page_number + 1
                )

                # Save the page number.
                source_pages.append(
                    page_number
                )


        # ====================================================
        # STEP 25: CREATE CONVERSATION HISTORY
        # ====================================================

        # Start with an empty conversation history.
        conversation_history = ""


        # Read all previous messages.
        #
        # The last message is the current question,
        # so [:-1] removes it from the history.
        for message in (
            st.session_state.messages[:-1]
        ):

            # Add the message role.
            conversation_history += (
                message["role"].upper()
                + ": "
            )

            # Add the message content.
            conversation_history += (
                message["content"]
                + "\n\n"
            )


        # ====================================================
        # STEP 26: CREATE GROUNDED RAG PROMPT
        # ====================================================

        # Create the prompt that will be sent to Llama.
        prompt = f"""
You are a helpful AI assistant that answers questions
using an uploaded PDF.

Follow these rules carefully:

1. Use only the information provided in the PDF context.

2. Do not use outside knowledge.

3. If the answer is not available in the PDF context,
   clearly say:

   "The information is not available in the uploaded PDF."

4. Do not invent, assume, or fabricate information.

5. Give a clear and detailed explanation.

6. Use headings and bullet points when they make
   the answer easier to understand.

7. If the PDF contains an example related to the question,
   explain that example.

8. Use the conversation history only to understand
   follow-up questions.

9. If the user asks a follow-up question using words such
   as "it", "this", "that", or "they", use the conversation
   history to understand what they are referring to.

PDF CONTEXT:
{context}

CONVERSATION HISTORY:
{conversation_history}

CURRENT USER QUESTION:
{question}

Now answer the user's question using only the PDF context.
"""


        # ====================================================
        # STEP 27: GENERATE ANSWER USING LLAMA
        # ====================================================

        # Show a loading message while Llama is working.
        with st.spinner(
            "🔎 Searching the PDF and generating an answer..."
        ):

            # Try to generate the answer.
            try:

                # Send the prompt to Llama.
                response = llm.invoke(
                    prompt
                )

                # Extract the generated text.
                answer = response.content


            # Handle Llama/Ollama errors.
            except Exception as error:

                # Create a user-friendly error answer.
                answer = (
                    "❌ An error occurred while generating "
                    "the answer. Please make sure Ollama "
                    "and the llama3.2 model are running."
                )

                # Display the technical error.
                st.error(
                    str(error)
                )


        # ====================================================
        # STEP 28: DISPLAY AI ANSWER
        # ====================================================

        # Create an assistant chat message.
        with st.chat_message(
            "assistant"
        ):

            # Display the AI answer.
            st.write(
                answer
            )


            # ------------------------------------------------
            # Display PDF source pages
            # ------------------------------------------------

            # Remove duplicate page numbers.
            unique_pages = sorted(
                set(source_pages)
            )


            # Show source pages if available.
            if unique_pages:

                # Create an expandable source section.
                with st.expander(
                    "📚 View PDF Sources"
                ):

                    # Explain the sources.
                    st.write(
                        "The answer was generated using "
                        "information retrieved from these "
                        "PDF pages:"
                    )


                    # Display every source page.
                    for page in unique_pages:

                        st.write(
                            f"📄 Page {page}"
                        )


            # ------------------------------------------------
            # Display retrieval scores
            # ------------------------------------------------

            # Create an expandable score section.
            with st.expander(
                "📊 View Retrieval Scores"
            ):

                # Explain what the scores mean.
                st.write(
                    "These are the raw FAISS retrieval "
                    "distance scores for the PDF chunks "
                    "used to generate the answer."
                )


                # Display every retrieved score.
                for i, score in enumerate(
                    similarity_scores
                ):

                    st.write(
                        f"📌 Chunk {i + 1} — "
                        f"Score: {score:.4f}"
                    )


        # ====================================================
        # STEP 29: SAVE AI ANSWER
        # ====================================================

        # Save the assistant's answer to chat history.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


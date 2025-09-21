import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import shutil

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="GenTune", layout="wide")
st.title("🎛️ GenTune")
st.write("Tune system prompts + Chroma context and test model outputs dynamically.")

MODEL_FOLDER = "./gen_model"
VECTOR_DB_DIR = os.path.join(MODEL_FOLDER, "chroma_db")
SYSTEM_PROMPT_FILE = os.path.join(MODEL_FOLDER, "system_prompt.txt")
os.makedirs(MODEL_FOLDER, exist_ok=True)

tab1, tab2 = st.tabs(["⚙️ Model Creation", "🧪 Model Testing"])

# -------------------------
# Tab 1: Model Creation
# -------------------------
with tab1:
    st.header("⚙️ Create / Persist Model")
    
    # Step 1: Prebuilt Information
    prebuilt_information = st.text_area(
        "Step 1: Prebuilt Information",
        placeholder="Enter your prebuilt information here...",
        height=150
    )

    # Step 2: Chunk Settings
    chunk_size = st.slider("Chunk Size", 50, 1000, 200, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50, 10)

    # Step 3: System Prompt
    system_prompt = st.text_area(
        "Step 3: System Prompt",
        placeholder="Enter System Prompt here...",
        height=150
    )

    # Initialize session state
    if "chunks" not in st.session_state:
        st.session_state.chunks = []

    # Step 4: Actions
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        create_chunks_clicked = st.button("Generate Chunks")
    with col2:
        create_vector_clicked = st.button("Create VectorDB")
    with col3:
        clear_model_clicked = st.button("Clear Model")

    # -------------------------
    # Generate Chunks
    # -------------------------
    if create_chunks_clicked:
        if not prebuilt_information.strip():
            st.warning("Please enter prebuilt information.")
        else:
            doc = Document(page_content=prebuilt_information.strip(), metadata={"source": "user_input"})
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents([doc])
            st.session_state.chunks = chunks
            st.success(f"✅ Text split into {len(chunks)} chunk(s).")

            # Scrollable preview
            all_chunks_html = ""
            for i, c in enumerate(chunks, start=1):
                all_chunks_html += f"<b>Chunk {i}:</b><br>{c.page_content}<br><hr>"

            st.markdown(
                f"<div style='height:300px; overflow-y:scroll; border:1px solid #ccc; padding:10px;'>{all_chunks_html}</div>",
                unsafe_allow_html=True
            )

# -------------------------
# Create VectorDB + Save System Prompt
# -------------------------
if create_vector_clicked:
    if not st.session_state.chunks:
        st.warning("Please generate chunks first before creating VectorDB.")
    else:
        try:
            # Clear previous VectorDB if it exists
            if os.path.exists(VECTOR_DB_DIR):
                shutil.rmtree(VECTOR_DB_DIR)
            os.makedirs(VECTOR_DB_DIR)

            # Create Chroma VectorDB
            embedding = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma.from_documents(
                documents=st.session_state.chunks,
                embedding=embedding,
                persist_directory=VECTOR_DB_DIR
            )
            st.success("✅ VectorDB created and persisted successfully! Previous database cleared.")

            # Save system prompt
            if system_prompt.strip():
                with open(SYSTEM_PROMPT_FILE, "w") as f:
                    f.write(system_prompt.strip())
                st.info("System prompt saved successfully!")
            else:
                st.warning("System prompt is empty; nothing was saved.")

        except PermissionError:
            st.error(f"❌ Cannot write to '{VECTOR_DB_DIR}'. Check folder permissions.")
        except Exception as e:
            st.error(f"❌ Failed to create VectorDB: {e}")







    

    # -------------------------
    # Clear Model
    # -------------------------
    if clear_model_clicked:
        if os.path.exists(MODEL_FOLDER):
            shutil.rmtree(MODEL_FOLDER)
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        st.session_state.chunks = []
        st.success("🗑️ Model cleared! You can start fresh.")

# -------------------------
# Tab 2: Model Testing
# -------------------------
with tab2:
    st.header("🧪 Test the Model")

    # Load system prompt
    try:
        with open(SYSTEM_PROMPT_FILE, "r") as f:
            loaded_system_prompt = f.read()
    except FileNotFoundError:
        loaded_system_prompt = ""

    if loaded_system_prompt:
        loaded_system_prompt = st.text_area("Loaded System Prompt (editable)", value=loaded_system_prompt, height=150)
    else:
        st.warning("No system prompt found. Create one first in the Model Creation tab.")

    user_input = st.text_area("User Input", placeholder="Type something to test...", height=100)
    top_k = st.slider("Number of chunks to retrieve for context", 1, 10, 3)

    if st.button("Run Model", key="run_model"):
        if not loaded_system_prompt:
            st.warning("System prompt not loaded. Please create model first.")
        elif not user_input:
            st.warning("Please enter some user input!")
        else:
            # Load VectorDB
            if not os.path.exists(VECTOR_DB_DIR):
                st.warning("VectorDB not found. Please create model first.")
            else:
                vectordb = Chroma(
                    persist_directory=VECTOR_DB_DIR,
                    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
                )

                # Retrieve top-k relevant chunks
                retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
                relevant_docs = retriever.get_relevant_documents(user_input)
                context_text = "\n".join([d.page_content for d in relevant_docs])

                # Build prompt dynamically
                llm = OllamaLLM(model="qwen2.5:1.5b")
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", loaded_system_prompt + "\nContext:\n{context}"),
                    ("user", "Question: {question}")
                ])
                parser = StrOutputParser()

                # Generate output
                prompt_input = {"context": context_text, "question": user_input}
                rag_chain = RunnablePassthrough() | prompt_template | llm | parser

                with st.spinner("Generating response..."):
                    response = rag_chain.invoke(prompt_input)

                # Display result
                st.success("Generated Response:")
                st.write(response)

                # Show relevant chunks
                with st.expander("Relevant Chunks Used for Answer"):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(f"**Chunk {i}:** {doc.page_content}")

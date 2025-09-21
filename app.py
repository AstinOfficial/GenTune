import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="GenTune")
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

    prebuilt_information = st.text_area(
        "Prebuilt Information :",
        placeholder="Enter your prebuilt information here...",
        height=150
    )
    chunk_size = st.slider("Chunk Size", min_value=50, max_value=1000, value=200, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)
    system_prompt = st.text_area(
        "System Prompt :",
        placeholder="Enter System Prompt here...",
        height=150
    )

    if "chunks" not in st.session_state:
        st.session_state.chunks = []

    col1, col2 = st.columns(2)
    with col1:
        create_chunks_clicked = st.button("Create Chunks")
    with col2:
        create_vector_clicked = st.button("Create VectorDB")

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
            st.success(f"Text split into {len(chunks)} chunk(s).")

            # Scrollable preview
            all_chunks_html = ""
            for i, c in enumerate(chunks, start=1):
                all_chunks_html += f"<b>Chunk {i}:</b><br>{c.page_content}<br><hr>"

            st.markdown(
                f"<div style='height:300px; overflow-y:scroll; border:1px solid #ccc; padding:10px;'>{all_chunks_html}</div>",
                unsafe_allow_html=True
            )

if create_vector_clicked:
    if not st.session_state.chunks:
        st.warning("Please generate chunks first before creating VectorDB.")
    else:
        try:
            # Ensure the folder exists
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)

            # Check if folder is writable
            test_file = os.path.join(VECTOR_DB_DIR, "test_write.txt")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)

            # Create Chroma VectorDB
            embedding = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma.from_documents(
                documents=st.session_state.chunks,
                embedding=embedding,
                persist_directory=VECTOR_DB_DIR
            )
            st.success("✅ VectorDB created and persisted successfully!")

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
# Tab 2: Model Testing
# -------------------------
with tab2:
    st.header("🧪 Test the Model")

    try:
        with open(SYSTEM_PROMPT_FILE, "r") as f:
            loaded_system_prompt = f.read()
    except FileNotFoundError:
        loaded_system_prompt = ""

    if loaded_system_prompt:
        st.info(f"Loaded system prompt:\n{loaded_system_prompt}")
    else:
        st.warning("No system prompt found. Create one first in the Model Creation tab.")

    user_input = st.text_area(
        "User Input",
        placeholder="Type something to test with your system prompt...",
        height=100
    )

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
                retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(user_input)
                context_text = "\n".join([d.page_content for d in relevant_docs])

                # Build prompt dynamically
                llm = OllamaLLM(model="qwen2.5:1.5b")
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", loaded_system_prompt + "\nContext:\n{context}"),
                    ("user", "Question: {question}")
                ])
                parser = StrOutputParser()

                # Prepare input
                prompt_input = {"context": context_text, "question": user_input}

                # Generate output
                rag_chain = RunnablePassthrough() | prompt_template | llm | parser
                response = rag_chain.invoke(prompt_input)

                # Display result
                st.success("Generated Response:")
                st.write(response)

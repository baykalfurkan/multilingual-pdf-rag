try:
    # Streamlit
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # Lokal
    pass
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langdetect import detect
import tempfile
import os
import re

load_dotenv()

# Streamlit UI 
st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")
st.title("📚 Smart Document Assistant")


if 'document_processed' not in st.session_state:
    st.session_state['document_processed'] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []

model_descriptions= {
    "gemini-1.5-pro": "The best performing model for large and complex PDF documents. Thanks to its long context window, it can understand the entire document, providing more consistent and comprehensive answers.",
    "gemini-1.5-flash": "If your priority is a fast response time, you can try this model. However, it may not perform as in-depth analysis as the 'pro' model on very long or detailed documents.",
    "gemini-1.5-flash-8b": "A suitable option if you are looking for quick answers to a high number of simple questions. Information loss may occur in large or complex documents.",
    "gemini-2.0-flash": "A new generation fast model. It is expected to perform well even in more complex queries in the future. Current tests show promising results.",
    "gemini-2.0-flash-lite": "Focused on cost-effectiveness and low latency. You can consider it especially when you are looking for short and concise answers or have budget constraints.",
    }


# Sidebar
with st.sidebar:
    
    selected_language = st.selectbox(
        "💬 Choose Conversation Language",
        ["English", "Turkish", "French", "German", "Spanish"],
        index=0
    )
    st.sidebar.header("🤖 Gemini Model Selection")
    model_options = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
    selected_model = st.selectbox(
        "Select Model",
        model_options,
        index=0,
        format_func=lambda x: x
    )

    st.sidebar.header("📊 Document Statistics")
    if 'document_stats' in st.session_state:
        stats = st.session_state['document_stats']
        st.sidebar.write(f"- Total Pages: {stats['total_pages']}")
        st.sidebar.write(f"- Total Characters: {stats['total_characters']:,}")
        st.sidebar.write(f"- Detected Language: {stats['detected_language'].upper()}")
    else:
        st.sidebar.write("No document uploaded yet.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Descriptions:")
    for model, description in model_descriptions.items():
        st.sidebar.write(f"**{model}:** {description}")


# PDF yuklenmesi ve islenmesi
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        data = loader.load()
        combined_text = " ".join([doc.page_content for doc in data[:5]])
        doc_language = detect(combined_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n##", "\n•", "\n-", "\n", ". ", " "],
            length_function=len,
            is_separator_regex=False
        )
        docs = text_splitter.split_documents(data)
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        st.session_state['document_stats'] = {
            'total_pages': len(data),
            'total_characters': sum(len(d.page_content) for d in docs),
            'detected_language': doc_language
        }
        st.session_state['processed_data'] = {
            'docs': docs,
            'doc_language': doc_language,
            'retriever': vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "lambda_mult": 0.6}
            ),
            'page_count': len(data)
        }
        st.session_state['document_processed'] = True
        return True

    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        return False
    finally:
        os.unlink(tmp_path)

# system prompt 
def create_system_prompt(target_lang, doc_lang, page_count):
    return f"""**Document Expert Assistant Rules**

1. **Core Principles:**
    - Provide answers only in {target_lang}
    - Keep technical terms in their original form
    - **Only use information directly and specifically related to the user's question.**
    - **Prioritize concise and direct answers.**

2. **Information Usage:**
    - Use only the provided content.
    - If the answer is not explicitly found in the document, say: "This information is not available in the document."
    - **Do not include irrelevant sentences or paragraphs, even if they mention keywords from the question.**

3. **Response Format:**
    - Organize as bullet points where appropriate.
    - Make important data **bold**.
    - Show formulas with LaTeX: $E=mc^2$

4. **Level of Detail:**
    - Provide short and direct answers for simple questions.
    - For complex questions, provide in-depth analysis **only if all the necessary information is directly available and related.**
    - Combine fragmented information **only if it directly answers the user's query.**

**Content:**
{{context}}"""


def validate_answer(answer, context):
    for doc in context:
        if doc.page_content.lower() in answer.lower():
            return True
    return "**Relevant information from the document:**" not in answer


uploaded_file = st.file_uploader("📤 Upload a PDF Document", type=["pdf"])

if uploaded_file:
    if process_pdf(uploaded_file):
        st.success(f"✅ Successfully uploaded document with {st.session_state['processed_data']['page_count']} pages! "
                         f"(Detected language: {st.session_state['processed_data']['doc_language'].upper()})")


if st.session_state['document_processed']:
    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.2,
        max_tokens=2000
    )

    query = st.chat_input("❓ Ask your question here...")

    if query:
        st.session_state.setdefault('query_attempted', True)
        st.session_state["messages"].append({"role": "user", "content": query})

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", create_system_prompt(
                selected_language,
                st.session_state['processed_data']['doc_language'],
                st.session_state['processed_data']['page_count']
            )),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(st.session_state['processed_data']['retriever'], document_chain)

        try:
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]

            if not validate_answer(answer, response["context"]):
                answer = "This information is not available in the document."

            st.session_state["messages"].append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Query processing error: {str(e)}")

# Sohbet gecmisini goster
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if not uploaded_file and st.session_state.get('query_attempted'):
    st.warning("⚠️ Please upload a document first!")

if uploaded_file and not st.session_state['document_processed']:
    st.info("⏳ Document is being processed. Please wait...")

 #           streamlit run "f:/Visual Studio Projeler/VS_Code/sda_env/app.py" 
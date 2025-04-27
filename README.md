# 📚 Smart Document Assistant

This project is an application that allows users to upload PDF documents and ask questions about the document content using the RAG approach with Google Generative AI (Gemini) models and Chroma vector database.

## 🚀 Features

- 📄 **PDF document upload and automatic pre-processing** 
- 📚 Text fragmentation with **LangChain-based RecursiveCharacterTextSplitter**
- 💾 Generating vectors using **GoogleGenerativeAIEmbeddings** 
- 🗄️ Persistent storage and MMR (Maximal Marginal Relevance) search with **Chroma vector database** 
- 🤖 **Various Gemini models**
- 🌐 **Multi-language support**: English, Turkish, French, German, Spanish
- 💬 **User-friendly interface**: Q&A demonstration with Streamlit chat UI

## 🌐 Deploy Link
- https://smart-document-assistant.streamlit.app/

## ⚙️ Requirements

- Git
- Python 3.9 
- Google API credentials (GOOGLE_API_KEY)

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/baykalfurkan/Smart-Document-Assistant.git
   cd smart-document-assistant
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Unix/macOS
   venv\\Scripts\\activate       # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file and add the necessary variables:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## ▶️  Running

Start the Streamlit application with the following command:
```bash
streamlit run app.py
```

On the page that opens in your browser, upload your PDF file and proceed to the question-and-answer section.

## 💡 Usage

1. **Select Conversation Language** 
2. Choose the Gemini model that suits your needs from the **Select Model** section.
3. Upload your document by clicking the Upload a **Upload a PDF Document** button.
4. After the document is processed, the number of pages, number of characters, and detected language will be displayed in the sidebar.
5. Get answers from the RAG assistant by typing your questions in the chat box.

## 📁 Project File Structure
```
smart-document-assistant/
├── app.py              # Main Streamlit application code
├── requirements.txt    # List of dependencies
├── .env                # .env file
└── chroma_db/          # Chroma database persistent folder
```

## 🤝 Contribution

For bug reports or feature suggestions, please open an issue or send a pull request.

## 📧 Contact
For questions: furkanbaykal001@gmail.com

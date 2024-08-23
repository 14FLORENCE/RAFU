from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from fpdf import FPDF
import os
from flask_cors import CORS
import openai

app = Flask(__name__, static_folder='assets')
CORS(app)

load_dotenv()

# Load environment variables for OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/upload', methods=['POST'])
def upload_content():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file!', 400

    # Save the uploaded PDF file
    pdf_path = os.path.join('uploads', file.filename)
    file.save(pdf_path)

    # Process and store content from the PDF
    process_pdf_content(pdf_path)

    return jsonify({'message': 'PDF uploaded successfully'})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_message = data.get('message', '')

    # Check for PDF-based responses
    pdf_response = extract_pdf_response(user_message)
    if pdf_response:
        return jsonify({'response': pdf_response, 'context': []})

    # Fallback to OpenAI API for a broader response
    openai.api_key = OPENAI_API_KEY
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=user_message,
            max_tokens=150
        )
        return jsonify({'response': response.choices[0].text.strip(), 'context': []})
    except Exception as e:
        print("Error calling OpenAI API:", e)
        return jsonify({'response': "I'm sorry, but I couldn't find an answer to that.", 'context': []})

def process_pdf_content(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vector_db")

def extract_pdf_response(query):
    retriever = FAISS.load_local("vector_db", OpenAIEmbeddings()).as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(ChatOpenAI(), hub.pull("langchain-ai/retrieval-qa-chat")))

    response = retrieval_chain.invoke({"input": query})
    return response.get("answer", "I couldn't find an answer to that in the provided document.")

def generate_content(content_type, custom_prompt, custom_prompt_result):
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatOpenAI()

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retriever = FAISS.load_local("vector_db", OpenAIEmbeddings()).as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    if custom_prompt_result:
        input_text = custom_prompt_result
    elif custom_prompt:
        input_text = custom_prompt
    else:
        input_texts = {
            'questions': (
                "Based on the content, generate 10 multiple choice questions with 4 answer options each, "
                "10 true or false questions, 5 definition questions, and 5 short notes questions."
            ),
            'answers': "Based on the content, generate detailed answers to key questions.",
            'lesson_plan': "Based on the content, generate a lesson plan and weekly timetable for a student to follow.",
            'summary': "Based on the content, generate a summary of the document."
        }
        input_text = input_texts.get(content_type, "Based on the content, generate a summary.")

    response = retrieval_chain.invoke({"input": input_text})
    return response["answer"]

def create_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf_filename = "generated_content.pdf"
    pdf.output(pdf_filename)
    return pdf_filename

if __name__ == "__main__":
    app.run(debug=True, port=7000)

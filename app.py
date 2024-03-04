import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings #permet d'accéder aux modèles d'embeddings génératifs de Google AI
import google.generativeai as genai #offre un accès direct aux modèles d'intelligence artificielle générative de Google
from langchain.vectorstores import FAISS #stocker et rechercher des vecteurs d'embeddings dans le cadre d'un pipeline Langchain
from langchain_google_genai import ChatGoogleGenerativeAI #permet d'accéder aux modèles conversationnels génératifs de Google AI depuis l'environnement Langchain
from langchain.chains.question_answering import load_qa_chain#permet de charger et instancier une chaîne de traitement pour la question-réponse dans le framework Langchain
from langchain.prompts import PromptTemplate#offre une approche structurée pour construire des prompts avec des variables dynamiques et des exemples de réponses.
from dotenv import load_dotenv #permet de charger des variables d'environnement à partir d'un fichi
from langchain_openai.chat_models import ChatOpenAI #Cette bibliothèque permet la création d'un modèle de conversation basé sur OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings #Cette bibliothèque permet d'utiliser des modèles d'embeddings OpenAI pour convertir du texte en vecteurs d'embeddings sémantiques
from pymongo import MongoClient

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/rag_system')
db = client['rag_system']  # Replace 'your_database_name' with your actual database name
users_collection = db['chunk_detail']

load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
   

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
  
    for chunk_index, chunk in enumerate(chunks):
        embeddings  = OpenAIEmbeddings()
        lines = chunk.split('\n')
        num_lines = len(lines)
        query_result = embeddings.embed_query(chunk)
        chunk_data = {
            'chunk_index': chunk_index,
            'chunk_text': chunk,
            'num_lines': num_lines,
            'lines': [{'line_index': index, 'line_text': line} for index, line in enumerate(lines)],
            'embedding':query_result
        }
        users_collection.insert_one(chunk_data)
        print("Inserted chunk", chunk_index, "into MongoDB")
        print("-" * 50)
    
    print("Number of chunks inserted into MongoDB:", len(chunks))  
    return chunks

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
  
#     for chunk in chunks:
#         lines = chunk.split('\n')
#         num_lines = len(lines)
#         print("Number of lines in this chunk:", num_lines)
    
#         print("-" * 50)
#     print("Number of chunks:",len(chunks)  )  
#     return chunks


def get_vector_store(text_chunks):
    embeddings  = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    llm  = ChatOpenAI(temperature=0 , model="gpt-3.5-turbo-1106")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm , chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    print(docs)
    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using open ai💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
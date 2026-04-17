import fitz
import base64
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
# IMPORTAÇÕES NOVAS (Substituem o antigo load_qa_chain)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def get_pdf_text(arquivos_info):
    docs = []
    vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    for nome_do_arquivo, pdf_bytes in arquivos_info:

        pdf_document = fitz.open(stream=pdf_bytes, filetype='pdf')

        for i, page in enumerate(pdf_document):
            text = page.get_text()

            if not text or len(text.strip()) < 50:
                print(f"Página {i + 1} de {nome_do_arquivo} parece ser escaneada. Pedindo para o Gemini ler a imagem...")

                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes('png')

                img_base64 = base64.b64encode(img_data).decode('utf-8')

                mensagem = HumanMessage(
                    content=[
                        {"type": "text", "text": "Extraia todo o texto legível desta imagem de documento. Retorne apenas o texto, sem comentários adicionais."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                )
                
                resposta_vision = vision_model.invoke([mensagem])
                text = resposta_vision.content

            if text and text.strip():
                docs.append(Document(page_content=text, metadata={
                    "page": i + 1,
                    "nome_arquivo": nome_do_arquivo
                    }
                ))
        pdf_document.close()
    return docs

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_vector_store(text_chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vector_store = FAISS.from_documents(text_chunks, embedding=embedding)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Você é um assistente da recepção. Responda à pergunta da forma mais detalhada possível usando APENAS o contexto fornecido. 
    Sempre informe o número da página de onde você extraiu a informação (exemplo: "Fonte: Página X").
    Se a resposta não estiver no contexto fornecido, diga: "Desculpe, essa informação não está nos editais que eu li.", não invente a resposta.\n\n
    Contexto do Edital:\n {context}\n
    Pergunta do Usuário: \n{question}\n
    Resposta:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # PADRÃO NOVO (LCEL) - Evita o erro de "memory"
    chain = prompt | model | StrOutputParser()
    return chain


# Apagar tudo para baixo
def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    new_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

    # Acha os pedaços do edital que falam sobre a pergunta
    docs = new_db.similarity_search(user_question)
    
    # Junta os textos encontrados para passar para o modelo
    contexto_junto = "\n\n".join([f"Arquivo: {doc.metadata.get('nome_arquivo', 'Desconhecido')} - Página {doc.metadata.get('page', 'Desconhecida')}:\n{doc.page_content}" for doc in docs])

    chain = get_conversational_chain()
    
    # Chama a nova corrente usando o método .invoke()
    response = chain.invoke({"context": contexto_junto, "question": user_question})

    st.write("🤖 **Resposta:** ", response)

def main():
    st.set_page_config(page_title="Chatbot da Recepção", layout="wide")
    st.header("Assistente de Editais da Recepção 📚")

    user_question = st.text_input("Faça uma pergunta sobre o edital")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu Lateral: ")
        pdf_docs = st.file_uploader("Arraste seus PDFs aqui", accept_multiple_files=True)

        if st.button("Treinar Robo"):
            with st.spinner("Lendo os arquivos..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Pronto")

    
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # for m in genai.list_models():
    #     print(m.name, m.supported_generation_methods)

if __name__ == "__main__":
    main()
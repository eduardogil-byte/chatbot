import os
import fitz
import base64
import re
import unicodedata
import concurrent.futures
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import SupabaseVectorStore

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def limpar_nome_arquivo(nome: str) -> str:
    nome_sem_acentos = ''.join(c for c in unicodedata.normalize('NFD', nome)
                               if unicodedata.category(c) != 'Mn')
    
    nome_limpo = re.sub(r'[^a-zA-Z0-9.\-]', '_', nome_sem_acentos)

    nome_limpo = re.sub(r'_+', '_', nome_limpo)

    return nome_limpo

def process_page_with_gemini(vision_model, img_base64):
    mensagem = HumanMessage(
        content=[
            {"type": "text", "text": "Extraia todo o texto legível desta imagem de documento. Retorne apenas o texto, sem comentários adicionais."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
    )

    resposta_vision = vision_model.invoke([mensagem])
    return resposta_vision.content

def get_pdf_text(arquivos_info):
    docs = []
    vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    for nome_do_arquivo, pdf_bytes in arquivos_info:
        pdf_document = fitz.open(stream=pdf_bytes, filetype='pdf')

        nome_do_arquivo = limpar_nome_arquivo(nome_do_arquivo)

        pages_to_ocr = []

        for i, page in enumerate(pdf_document):
            text = page.get_text()

            if text and len(text.strip()) >= 50:
                docs.append(Document(page_content=text, metadata={"page": i + 1, "nome_arquivo": nome_do_arquivo}))
            else:
                print(f"Página {i + 1} de {nome_do_arquivo} parece ser escaneada. Pedindo para o Gemini ler a imagem...")
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes('png')
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                pages_to_ocr.append((i + 1, img_base64))
                
        if pages_to_ocr:
            # aqui pode colocar os workers entre 5 ou mais 
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_page = {executor.submit(process_page_with_gemini, vision_model, img_base64): page_num for page_num, img_base64 in pages_to_ocr}

                for future in concurrent.futures.as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        extracted_text = future.result()
                        if extracted_text and extracted_text.strip():
                            docs.append(Document(page_content=extracted_text, metadata={"page": page_num, "nome_arquivo": nome_do_arquivo}))
                    except Exception as e:
                        print(f"Erro ao processar página {page_num}: {e}")
        pdf_document.close()

        docs.sort(key=lambda doc: (doc.metadata.get("nome_arquivo", ""), doc.metadata.get("page", 0)))

    return docs

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_vector_store(text_chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    SupabaseVectorStore.from_documents(
        documents=text_chunks,
        embedding=embedding,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )

def get_conversational_chain():
    prompt_template = """
    Você é um assistente institucional especializado em responder perguntas sobre editais.

    Responda usando APENAS as informações do contexto fornecido.
    Se a informação não estiver no contexto, diga claramente que não encontrou essa informação no edital.

    Regras:
    - Responda de forma direta primeiro.
    - Depois explique detalhes importantes, se existirem.
    - Se houver datas, destaque as datas.
    - Se houver prazo inicial e final, informe os dois.
    - Se a pergunta estiver ambígua, explique o que foi encontrado relacionado ao assunto.
    - Não invente informações.
    - No final, cite a fonte com nome do arquivo e página.
    - Se existirem vários editais no contexto, diga de qual edital veio cada informação.

    Contexto:
    {context}

    Pergunta do usuário:
    {question}

    Resposta:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    chain = prompt | model | StrOutputParser()
    return chain

def formatar_resposta(texto: str):
    texto = texto.replace("\r\n", "\n")

    # Remove espaços desnecessários
    texto = re.sub(r'[ \t]+', ' ', texto)
    texto = re.sub(r' *\n *', '\n', texto)

    # Garante quebra antes de listas numeradas: 1. 2. 3.
    texto = re.sub(r'\n(?=\d+\.\s)', '\n\n', texto)

    # Garante quebra antes de bullet points
    texto = re.sub(r'\n(?=[\-•]\s)', '\n\n', texto)

    # Garante quebra depois de linhas em negrito que funcionam como título
    texto = re.sub(r'(\*\*[^*\n]+:\*\*)\n', r'\1\n\n', texto)

    # Garante quebra antes de "Fonte:"
    texto = re.sub(r'\n(?=Fonte:)', '\n\n', texto)
    texto = re.sub(r'\n(?=\*\*Fonte)', '\n\n', texto)

    # Evita excesso de quebras
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    return texto.strip()
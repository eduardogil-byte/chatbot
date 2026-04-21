import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from app import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain,
    limpar_nome_arquivo
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="API - Chatbot", description="API para Gestão de Conhecimento Institucional")

# --- ADICIONE ESTE BLOCO ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite que qualquer origem acesse a API (ideal para testes locais)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PerguntaRequest(BaseModel):
    pergunta: str
    arquivo_escolhido: str | None=None

class RespostaResponse(BaseModel):
    resposta: str

@app.get('/')
def inicio():
    return {"status":200}

@app.delete("/arquivos/{nome_arquivo}")
async def excluir_arquivo(nome_arquivo: str):
    try:
        supabase.table("documents").delete().filter("metadata->>nome_arquivo", "eq", nome_arquivo).execute()

        supabase.storage.from_("editais").remove([nome_arquivo])

        return {"status": "sucesso", "mensagem": f"Arquivo {nome_arquivo} removido com sucesso."}

    except Exception as e:
        print(f"Erro ao excluir: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/arquivos', summary="Listar arquivos disponiveis na base")
async def listar_arquivos():
    try:
        resposta = supabase.table('documents').select('metadata').execute()

        nomes_arquivos = set()
        for item in resposta.data:
            metadata = item.get('metadata', {})
            nome = metadata.get('nome_arquivo')
            if nome:
                nomes_arquivos.add(nome)

        return {'arquivos': sorted(list(nomes_arquivos))}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao listar arquivos: {str(e)}")


@app.post("/treinar", summary="API para Gestão de Conhecimento Institucional")
async def treinar_base(arquivos: list[UploadFile] = File(...)):
    try:
        arquivos_info = []
        for file in arquivos:
            nome_seguro = limpar_nome_arquivo(file.filename)
            conteudo = await file.read()
            arquivos_info.append((nome_seguro, conteudo))

            try:
                supabase.storage.from_('editais').upload(
                    path=nome_seguro,
                    file=conteudo,
                    file_options={"content-type": "application/pdf"}
                )
            except Exception as e:
                print(f"Aviso: Arquivo {nome_seguro} já existe no storage ou erro ao salvar: {e}")

        raw_text = get_pdf_text(arquivos_info)
        if not raw_text:
            raise HTTPException(status_code=400, detail="Nenhum texto extraído.")

        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        return {"mensagem": f"Base treinada com sucesso usando {len(arquivos)} arquivo(s)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# @app.post('/perguntar', response_model=RespostaResponse, summary="Fazer uma pergunta sobre os documentos processados")
# async def fazer_pergunta(request: PerguntaRequest):
#     try:
#         embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
#         vector_store = SupabaseVectorStore(
#             embedding=embedding,
#             client=supabase,
#             table_name="documents",
#             query_name="match_documents"
#         )

#         docs = vector_store.similarity_search(request.pergunta)

#         print(f"docs: {docs}")

#         if not docs:
#             return RespostaResponse(resposta="Desculpe, não encontrei informações relevantes na base.")

#         contexto_junto = "\n\n".join([f"Arquivo: {doc.metadata.get('nome_arquivo', 'Desconhecido')} - Página {doc.metadata.get('page', 'Desconhecida')}:\n{doc.page_content}" for doc in docs])

#         chain = get_conversational_chain()
#         response = chain.invoke({"context": contexto_junto, "question": request.pergunta})

#         return RespostaResponse(resposta=response)
    
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Erro ao processar pergunta: {str(e)}")

@app.post('/perguntar', response_model=RespostaResponse, summary="Fazer uma pergunta sobre os documentos processados")
async def fazer_pergunta(request: PerguntaRequest):
    try:
        # 1. Transforma a pergunta do usuário em um vetor matemático
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vetor_pergunta = embedding_model.embed_query(request.pergunta)
        
        filtro_banco = {}
        if request.arquivo_escolhido and request.arquivo_escolhido != "Todos":
            filtro_banco = {"nome_arquivo": request.arquivo_escolhido}

        resposta_banco = supabase.rpc(
            "match_documents",
            {
                "query_embedding": vetor_pergunta,
                "match_count": 4,
                 "filter": filtro_banco
            }
        ).execute()

        # 3. Converte o resultado de volta para o formato que o seu código já usa (Document)
        from langchain_core.documents import Document
        docs = []
        for item in resposta_banco.data:
            docs.append(Document(
                page_content=item.get("content", ""),
                metadata=item.get("metadata", {})
            ))

        print(f"Trechos encontrados no banco: {len(docs)}")

        if not docs:
            return RespostaResponse(resposta="Desculpe, não encontrei informações relevantes na base.")

        # 4. Junta os textos encontrados e manda para o Gemini ler e responder
        contexto_junto = "\n\n".join([f"Arquivo: {doc.metadata.get('nome_arquivo', 'Desconhecido')} - Página {doc.metadata.get('page', 'Desconhecida')}:\n{doc.page_content}" for doc in docs])

        chain = get_conversational_chain()
        response = chain.invoke({"context": contexto_junto, "question": request.pergunta})

        return RespostaResponse(resposta=response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar pergunta: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
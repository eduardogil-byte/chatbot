import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain
)

app = FastAPI(title="API - Chatbot", description="API para Gestão de Conhecimento Institucional")

class PerguntaRequest(BaseModel):
    pergunta: str

class RespostaResponse(BaseModel):
    resposta: str

@app.post("/treinar", summary="API para Gestão de Conhecimento Institucional")
async def treinar_base(arquivos: list[UploadFile] = File(...)):
    try:
        arquivos_info = []
        for file in arquivos:
            conteudo = await file.read()
            arquivos_info.append((file.filename, conteudo))

        raw_text = get_pdf_text(arquivos_info)

        if not raw_text:
            raise HTTPException(status_code=400, detail="Nenhum texto pôde ser extraído dos arquivos enviados.")

        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        return {"mensagem": f"Base treinada com sucesso usando {len(arquivos)} arquivo(s)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post('/perguntar', response_model=RespostaResponse, summary="Fazer uma pergunta sobre os documentos processados")
async def fazer_pergunta(request: PerguntaRequest):
    if not os.path.exists("faiss_index"):
        raise HTTPException(status_code=400, detail="O índice do banco de dados não foi encontrado. Por favor, treine a base primeiro enviando os PDFs.")

    try:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(request.pergunta)

        contexto_junto = "\n\n".join([f"Arquivo: {doc.metadata.get('nome_arquivo', 'Desconhecido')} - Página {doc.metadata.get('page', 'Desconhecida')}:\n{doc.page_content}" for doc in docs])

        chain = get_conversational_chain()
        response = chain.invoke({"context": contexto_junto, "question": request.pergunta})

        return RespostaResponse(resposta=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar pergunta: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
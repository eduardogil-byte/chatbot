# 🤖 Chatbot com IA para consulta de documentos — Backend

Backend de um chatbot desenvolvido para responder perguntas com base em documentos PDF utilizando **Inteligência Artificial e RAG (Retrieval-Augmented Generation)**.

A aplicação processa documentos, gera embeddings, armazena os conteúdos em uma base vetorial e recupera os trechos mais relevantes para fornecer contexto ao modelo de IA.

Este repositório contém o **backend** do projeto.

🔗 **Frontend:**
https://github.com/eduardogil-byte/Front-Chatbot

---

## 📌 Sobre o projeto

O objetivo do projeto é permitir que usuários adicionem documentos PDF a uma base de conhecimento e façam perguntas sobre o conteúdo desses arquivos.

Quando uma pergunta é realizada, o sistema procura os trechos mais relevantes nos documentos e os utiliza como contexto para gerar uma resposta.

Além da resposta, a API informa **quais documentos e páginas foram utilizados como fonte**.

---

## 🧠 Como funciona

O projeto utiliza uma arquitetura baseada em **RAG — Retrieval-Augmented Generation**.

O fluxo principal funciona da seguinte forma:

```text
PDF
 ↓
Extração do texto
 ↓
Divisão em trechos
 ↓
Geração de embeddings
 ↓
Supabase / Banco Vetorial
 ↓
Busca por similaridade
 ↓
Contexto relevante
 ↓
Gemini
 ↓
Resposta + fontes
```

### 1. Processamento dos documentos

Os arquivos PDF enviados pelo usuário são processados página por página.

Quando a página possui texto, ele é extraído diretamente.

Caso seja uma página digitalizada ou com pouco texto detectável, o sistema transforma a página em imagem e utiliza o **Gemini** para realizar a leitura do conteúdo.

### 2. Divisão do conteúdo

O texto extraído é dividido em pequenos trechos utilizando o LangChain.

Isso permite que o sistema procure apenas as partes mais relevantes para cada pergunta.

### 3. Geração de embeddings

Cada trecho é convertido em uma representação vetorial utilizando:

```text
gemini-embedding-001
```

Esses vetores permitem realizar buscas por similaridade semântica.

### 4. Armazenamento

Os documentos processados e seus embeddings são armazenados no **Supabase**.

Também são mantidos metadados como:

* nome do arquivo;
* página de origem;
* conteúdo do trecho.

Os PDFs originais também são armazenados no Supabase Storage.

### 5. Consulta

Quando o usuário envia uma pergunta:

1. A pergunta é transformada em um embedding.
2. O Supabase procura os trechos semanticamente mais próximos.
3. Os trechos encontrados são reunidos como contexto.
4. O contexto e a pergunta são enviados ao Gemini.
5. O modelo gera a resposta utilizando os documentos encontrados.
6. A API retorna a resposta e as fontes utilizadas.

---

## ✨ Funcionalidades

* 📄 Upload de múltiplos arquivos PDF
* 🧠 Processamento de documentos com IA
* 🔎 Busca semântica por embeddings
* 📚 RAG para geração de respostas baseadas nos documentos
* 🖼️ Leitura de PDFs digitalizados utilizando Gemini
* 📑 Identificação das páginas utilizadas
* 🔗 Retorno das fontes das respostas
* 🎯 Consulta em um documento específico
* 🌐 Consulta em todos os documentos disponíveis
* 📂 Listagem dos documentos cadastrados
* 🗑️ Exclusão de documentos
* 🔒 Prevenção de documentos duplicados
* ☁️ Armazenamento de PDFs no Supabase Storage
* 🚀 API REST utilizando FastAPI

---

## 🛠️ Tecnologias utilizadas

* **Python**
* **FastAPI**
* **LangChain**
* **Google Gemini**
* **Gemini Embeddings**
* **Supabase**
* **Supabase Vector Store**
* **PyMuPDF**
* **Pydantic**
* **Uvicorn**

---

## 🤖 Modelos utilizados

### Geração de respostas

```text
Gemini 2.5 Flash
```

Utilizado para interpretar o contexto recuperado e gerar as respostas do chatbot.

### Embeddings

```text
gemini-embedding-001
```

Utilizado para transformar documentos e perguntas em vetores para busca semântica.

O Gemini também é utilizado para extrair conteúdo de páginas de PDF que não possuem texto diretamente acessível.

---

## 🌐 Principais endpoints

### Status da API

```http
GET /
```

Verifica se a API está disponível.

---

### Adicionar documentos

```http
POST /treinar
```

Recebe um ou mais arquivos PDF, processa o conteúdo e adiciona os documentos à base de conhecimento.

---

### Fazer uma pergunta

```http
POST /perguntar
```

Exemplo de requisição:

```json
{
  "pergunta": "Qual é o período de inscrição?",
  "arquivo_escolhido": "edital.pdf"
}
```

A API realiza a busca semântica e retorna a resposta junto das fontes utilizadas.

Exemplo simplificado:

```json
{
  "resposta": "O período de inscrição é...",
  "fontes": [
    {
      "arquivo": "edital.pdf",
      "paginas": [2, 3],
      "url": "/arquivos/edital.pdf/ver"
    }
  ]
}
```

---

### Listar documentos

```http
GET /arquivos
```

Retorna os arquivos disponíveis na base de conhecimento.

---

### Visualizar documento

```http
GET /arquivos/{nome_arquivo}/ver
```

Redireciona para o PDF armazenado.

---

### Remover documento

```http
DELETE /arquivos/{nome_arquivo}
```

Remove o documento e seus dados associados.

---

## ⚙️ Executando o projeto

### Pré-requisitos

Tenha instalado:

* Python 3
* Conta no Google AI / Gemini
* Projeto configurado no Supabase

---

### Clone o repositório

```bash
git clone https://github.com/eduardogil-byte/chatbot.git
```

Entre na pasta:

```bash
cd chatbot
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 🔐 Variáveis de ambiente

Crie um arquivo `.env` baseado no `.env.example`:

```env
GOOGLE_API_KEY=sua_chave_google
SUPABASE_URL=sua_url_supabase
SUPABASE_KEY=sua_chave_supabase
```

Não envie suas chaves reais para o repositório.

---

## ▶️ Iniciando a API

Execute:

```bash
uvicorn api:app --reload
```

A API ficará disponível normalmente em:

```text
http://localhost:8000
```

A documentação automática do FastAPI pode ser acessada em:

```text
http://localhost:8000/docs
```

---

## 📁 Estrutura do projeto

```text
chatbot/
│
├── api.py
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### `api.py`

Responsável pelos endpoints da API, integração com o Supabase e gerenciamento das requisições.

### `app.py`

Contém a lógica principal de processamento dos documentos, extração de texto, embeddings, busca vetorial e integração com o Gemini.

---

## 🔗 Frontend

A interface utilizada para consumir esta API está disponível em outro repositório:

**Front-Chatbot**

https://github.com/eduardogil-byte/Front-Chatbot

O frontend foi desenvolvido utilizando React e se comunica com esta API para upload dos documentos, consultas e gerenciamento da base de conhecimento.

---

## 🎯 Conceitos aplicados

Durante o desenvolvimento deste projeto foram aplicados conceitos como:

* APIs REST;
* Inteligência Artificial generativa;
* RAG;
* embeddings;
* busca vetorial;
* processamento de PDFs;
* OCR com IA;
* integração com serviços externos;
* bancos vetoriais;
* armazenamento em nuvem;
* processamento concorrente;
* integração frontend/backend.

---

## 👨‍💻 Autor

**Eduardo Machado Gil**

GitHub: [@eduardogil-byte](https://github.com/eduardogil-byte)

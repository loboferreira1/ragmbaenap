# RAG - Projeto MBA ENAP

## Aluno José Lobo

Este projeto é uma aplicação simples de Recuperação de Respostas (RAG) utilizando Streamlit, LangChain e OpenAI. Ele permite que os usuários façam upload de documentos PDF e façam perguntas sobre o conteúdo do documento.

### Funcionalidades
- Upload de documentos PDF.
- Divisão de texto em partes menores para processamento eficiente.
- Criação de embeddings usando OpenAI.
- Armazenamento vetorial com FAISS para recuperação eficiente.
- Interface de chat para perguntas e respostas baseadas no documento.

### Como usar
1. Configure a chave API da OpenAI:
   - Localmente: Crie um arquivo `.env` com a variável `OPENAI_API_KEY`.
   - Na nuvem Streamlit: Configure a chave em Configurações > Segredos.
2. Instale as dependências listadas no arquivo `requirements.txt`.
3. Execute o aplicativo Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Faça upload de um documento PDF e insira suas perguntas na interface.

### Dependências
As dependências do projeto estão listadas no arquivo `requirements.txt` e incluem:
- streamlit
- langchain
- langchain-community
- openai
- python-dotenv
- pypdf
- tiktoken
- faiss-cpu
- langchain-openai

### Estrutura do Projeto
- `app.py`: Código principal do aplicativo.
- `requirements.txt`: Lista de dependências do projeto.

### Autor
José Lobo

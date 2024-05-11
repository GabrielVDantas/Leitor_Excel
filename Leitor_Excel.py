import pandas as pd
import numpy as np
import google.generativeai as genai

from google.colab import userdata

API_KEY = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key = API_KEY)

# 1. Preparar os dados (Insira um arquivo .xlsx aqui, 'Pasta1.xlsx' é o nome da planilha de testes que eu estava usando)
df = pd.read_excel("Pasta1.xlsx")

# 2. Definir o modelo de embedding
model = "models/embedding-001"

# 3. Criar embeddings para todas as colunas da planilha
for coluna in df.columns:
    df[f"{coluna}_Embeddings"] = df[coluna].apply(lambda texto: genai.embed_content(
        model=model, content=str(texto), task_type="RETRIEVAL_QUERY"
    )["embedding"])

# 4. Função que busca as informações dentro da planilha
def busca_informacao_na_planilha(consulta, base, model):
    embedding_da_consulta = genai.embed_content(
        model=model, content=consulta, task_type="RETRIEVAL_QUERY"
    )["embedding"]

    # Palavras-chave para identificar a coluna de resposta
    palavras_chave = {
        "ticket": "NÚMERO DO TICKET",
        "solicitante": "SOLICITANTE",
        "categoria": "CATEGORIA",
        "urgência": "URGÊNCIA",
        "responsável": "RESPONSÁVEL"
    }

    # Encontrar a coluna de resposta basedada nas palavras-chave na consulta
    coluna_resposta = None
    for palavra, coluna in palavras_chave.items():
        if palavra in consulta.lower():
            coluna_resposta = coluna
            break

    if coluna_resposta is None:
        return "Não consegui entender a pergunta."

    # Calcular similaridade apenas para a coluna de resposta
    produtos_escalares = np.dot(np.stack(base[f"{coluna_resposta}_Embeddings"]), embedding_da_consulta)
    indice_linha_mais_similar = np.argmax(produtos_escalares)
    resposta = base.iloc[indice_linha_mais_similar][coluna_resposta]
    return resposta

# 5. Escreva uma consulta relacionada sua planilha
consulta = "Quantos tickets estão disponíveis?"
resultado = busca_informacao_na_planilha(consulta, df, model)

# 6. O resultado aparece na tela
print(resultado)
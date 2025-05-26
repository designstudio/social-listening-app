# app.py

import streamlit as st
import pandas as pd
import json
import io
from docx import Document
from youtube_comment_downloader import YoutubeCommentDownloader
import re
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
import networkx as nx
import os # Importar para acessar variáveis de ambiente

# Importar o Gemini SDK
import google.generativeai as genai

# --- Configuração da API do Gemini ---
# A chave da API deve ser definida como uma variável de ambiente no Streamlit Cloud
# ou diretamente no secrets.toml (método preferido para produção)
# Para testes locais, você pode definir GOOGLE_API_KEY no seu ambiente ou aqui (NÃO RECOMENDADO PARA PRODUÇÃO)
# st.secrets é a forma recomendada para Streamlit Cloud
gemini_api_key = st.secrets.get("GOOGLE_API_KEY")

if not gemini_api_key:
    st.error("A chave da API do Google Gemini não foi encontrada. "
             "Por favor, configure-a no `secrets.toml` ou como variável de ambiente GOOGLE_API_KEY.")
    st.stop()

genai.configure(api_key=gemini_api_key)

# Inicializar modelo Gemini 1.5 Flash (ou Gemini 1.0 Pro, dependendo da sua preferência e cota)
# Usando 1.5 Flash para maior velocidade
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Funções de Extração de Dados ---

@st.cache_data(show_spinner=False)
def extract_text_from_file(file_contents, file_extension):
    """
    Extrai texto de conteúdo de arquivo binário com base na extensão.
    Retorna uma LISTA de strings (linhas ou comentários).
    """
    text_content_list = []
    try:
        if file_extension == '.csv':
            df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
            if 'comentario' in df.columns:
                text_content_list = df['comentario'].dropna().astype(str).tolist()
            else:
                for _, row in df.iterrows():
                    text_content_list.append(" ".join(row.dropna().astype(str).tolist()))
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(io.BytesIO(file_contents))
            if 'comentario' in df.columns:
                text_content_list = df['comentario'].dropna().astype(str).tolist()
            else:
                for _, row in df.iterrows():
                    text_content_list.append(" ".join(row.dropna().astype(str).tolist()))
        elif file_extension in ['.doc', '.docx']:
            document = Document(io.BytesIO(file_contents))
            for para in document.paragraphs:
                if para.text.strip():
                    text_content_list.append(para.text)
        else:
            st.warning(f"Formato de arquivo não suportado: {file_extension}. Por favor, use CSV, XLS, XLSX, DOC ou DOCX.")
            return []
    except Exception as e:
        st.error(f"Erro ao extrair texto do arquivo: {e}")
        return []
    return text_content_list

@st.cache_data(show_spinner=False)
def download_youtube_comments(youtube_url):
    """
    Baixa comentários de um vídeo do YouTube.
    Retorna uma LISTA de strings, cada string sendo um comentário.
    """
    MAX_COMMENTS_LIMIT = 2000 # Definir o limite máximo desejado

    try:
        downloader = YoutubeCommentDownloader()
        video_id = None
        match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
        if match:
            video_id = match.group(1)

        if not video_id:
            st.error(f"URL do YouTube inválida: {youtube_url}. Por favor, insira uma URL de vídeo válida.")
            return []

        with st.spinner(f"Baixando comentários do vídeo: {youtube_url}..."):
            all_comments = []
            comment_count = 0

            try:
                comments_generator = downloader.get_comments(video_id)

                for comment in comments_generator:
                    all_comments.append(comment['text'])
                    comment_count += 1
                    if comment_count >= MAX_COMMENTS_LIMIT:
                        st.info(f"Limite de {MAX_COMMENTS_LIMIT} comentários atingido para o vídeo: {youtube_url}.")
                        break
            except Exception as e:
                st.error(f"Não foi possível baixar comentários do vídeo '{youtube_url}' (pode ser vídeo privado/sem comentários/problema de conexão): {e}")
                return []

            if not all_comments:
                st.warning(f"Não foram encontrados comentários para o vídeo: {youtube_url}.")
                return []

            st.success(f"Baixados {len(all_comments)} comentários do vídeo do YouTube '{youtube_url}'.")
            return all_comments
    except Exception as e:
        st.error(f"Erro geral ao baixar comentários do YouTube para '{youtube_url}': {e}. Verifique se a URL está correta ou se o vídeo tem comentários.")
        return []

# --- Funções de Análise com Gemini ---

@st.cache_data(show_spinner=True)
def analyze_text_with_gemini(text_to_analyze):
    if not text_to_analyze.strip():
        return {
            "sentiment": {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "no_sentiment_detected": 100.0},
            "topics": [],
            "term_clusters": {},
            "topic_relations": []
        }

    prompt = f"""
Analise o texto de comentários de redes sociais abaixo de forma **estritamente objetiva, factual e consistente**.
Extraia as informações solicitadas. **Calcule as porcentagens e contagens EXATAS** com base no total de comentários relevantes.

Forneça as seguintes informações em formato JSON, exatamente como a estrutura definida. **Não inclua nenhum texto adicional antes ou depois do JSON.**

1.  **Sentimento Geral:** A porcentagem de comentários classificados como 'Positivo', 'Neutro', 'Negativo' e 'Sem Sentimento Detectado'. As porcentagens devem somar 100%.
2.  **Temas Mais Citados:** Uma lista dos 5 a 10 temas PRINCIPAIS, RELEVANTES e DISTINTOS discutidos nos comentários. Para cada tema, forneça a contagem EXATA de comentários Positivos, Neutros e Negativos relacionados a ele. **Seja específico e conciso nos nomes dos temas** (ex: "Aposentadoria para MEI's", "Emissão de Notas Fiscais", "Regularização CNPJ"). **Evite temas muito genéricos** e garanta que sejam mutuamente exclusivos quando possível.
3.  **Agrupamento de Termos/Nuvem de Palavras:** Uma lista dos 10 a 20 termos ou palavras-chave **mais frequentes e significativos**. Para cada termo, forneça sua frequência (contagem de ocorrências). **Priorize substantivos, verbos ou adjetivos que realmente representem o conteúdo**. **NÃO inclua termos como 'positivo', 'negativo', 'neutro' ou palavras de parada comuns** que não adicionam valor temático (ex: "o", "a", "de", "e", "que"). Garanta uma boa VARIEDADE de termos.
4.  **Relação entre Temas:** Identifique pelo menos 3 a 5 pares de temas que frequentemente aparecem juntos nos comentários, indicando uma **relação clara e lógica**. Forneça uma breve e direta descrição da relação.

O JSON deve ter a seguinte estrutura (atenção aos tipos de dados e nomes das chaves):
```json
{{
  "sentiment": {{
    "positive": float,
    "neutral": float,
    "negative": float,
    "no_sentiment_detected": float
  }},
  "topics": [
    {{
      "name": "Nome do Tema",
      "positive": int,
      "neutral": int,
      "negative": int
    }}
  ],
  "term_clusters": {{
    "termo1": int,
    "termo2": int
  }},
  "topic_relations": [
    {{
      "source": "Tema A",
      "target": "Tema B",
      "description": "Breve descrição da relação"
    }}
  ]
}}
Texto para análise:
"{text_to_analyze}"
"""

try:
    response = model.generate_content(prompt)
    response_text = response.text.strip()

    # Limpar blocos de código JSON, se existirem
    if response_text.startswith("```json"):
        response_text = response_text[len("```json"):].strip()
    if response_text.endswith("```"):
        response_text = response_text[:-len("```")].strip()

    data = json.loads(response_text)

    # Garantir que no_sentiment_detected exista e que a soma seja 100%
    if 'no_sentiment_detected' not in data['sentiment']:
        total_sentimento_detectado = data['sentiment'].get('positive', 0) + data['sentiment'].get('neutral', 0) + data['sentiment'].get('negative', 0)
        data['sentiment']['no_sentiment_detected'] = round(100.0 - total_sentimento_detectado, 2)

    # Ajustar para garantir que a soma de todos os sentimentos seja 100
    total_sum = data['sentiment']['positive'] + data['sentiment']['neutral'] + data['sentiment']['negative'] + data['sentiment']['no_sentiment_detected']
    if total_sum != 100 and total_sum != 0:
        for key in data['sentiment']:
            data['sentiment'][key] = round(data['sentiment'][key] / total_sum * 100, 2)

    return data
except json.JSONDecodeError as e:
    st.error(f"Erro ao decodificar JSON da resposta do Gemini na análise principal: {e}")
    st.code(f"Resposta bruta do Gemini (Análise): {response_text}")
    return None
except Exception as e:
    st.error(f"Erro inesperado ao analisar com Gemini: {e}")
    return None
@st.cache_data(show_spinner=True)
def generate_qualitative_analysis(analysis_results):
sentiment = analysis_results.get('sentiment', {})
topics = analysis_results.get('topics', [])
term_clusters = analysis_results.get('term_clusters', {})
topic_relations = analysis_results.get('topic_relations', [])

# Formata os dados JSON ANTES de inseri-los na f-string
sentiment_json = json.dumps(sentiment, ensure_ascii=False)
topics_json = json.dumps(topics, ensure_ascii=False)
term_clusters_json = json.dumps(term_clusters, ensure_ascii=False)
topic_relations_json = json.dumps(topic_relations, ensure_ascii=False)

prompt = f"""
Com base na análise de social listening dos comentários de redes sociais fornecidos, atue como um especialista em Marketing de Produto, Social Listening e Data Analysis. Redija uma análise qualitativa abrangente em até 4 parágrafos, focando nos aprendizados e insights estratégicos.

Considere as seguintes informações:

Sentimento Geral: {sentiment_json}
Temas Mais Citados: {topics_json}
Agrupamento de Termos: {term_clusters_json}
Relação entre Temas: {topic_relations_json}
Sua análise deve:

Contextualizar o cenário geral do sentimento.

Destacar os temas mais relevantes e a percepção do público sobre eles.

Pontuar termos-chave e como eles refletem o engajamento ou preocupações.

Identificar oportunidades ou desafios emergentes a partir das relações entre temas.

Oferecer insights acionáveis para estratégias de produto, comunicação ou atendimento ao cliente.

Ser concisa e focada nos aprendizados mais importantes.
"""

try:
response = model.generate_content(prompt)
return response.text.strip()
except Exception as e:
st.error(f"Erro ao gerar análise qualitativa com Gemini: {e}")
return "Não foi possível gerar a análise qualitativa."

@st.cache_data(show_spinner=True)
def generate_persona_insights(analysis_results, original_text_sample):
sentiment = analysis_results.get('sentiment', {})
topics = analysis_results.get('topics', [])
term_clusters = analysis_results.get('term_clusters', {})

original_text_display = original_text_sample[:1000] + "..." if len(original_text_sample) > 1000 else original_text_sample

# Formata os dados JSON ANTES de inseri-los na f-string
sentiment_json = json.dumps(sentiment, ensure_ascii=False)
topics_json = json.dumps(topics, ensure_ascii=False)
term_clusters_json = json.dumps(term_clusters, ensure_ascii=False)

combined_context = f"""
Comentários originais (amostra): {original_text_display}
Resultados da análise:

Sentimento Geral: {sentiment_json}

Temas Mais Citados: {topics_json}

Agrupamento de Termos: {term_clusters_json}
"""

persona_prompt = f"""
Considerando a análise de social listening de um público X e o conceito de "personas sintéticas" (que buscam capturar características de um público com base em dados de comportamento e sentimentos), crie um insight de persona sintética para o público que gerou esses comentários.
Concentre-se em:

Quais são as principais dores e necessidades desse público?

Quais são seus interesses ou paixões (explícitas ou implícitas nos temas e termos)?

Qual é o tone geral da sua comunicação (positivo, negativo, neutro, impaciente, engajado, etc.)?

Quais são as oportunidades para engajar ou atender melhor essa persona?

Dê um nome sugestivo a essa persona sintética, como "O Crítico Construtivo" ou "A Consumidora Engajada".

Use até 3-4 parágrafos para descrever essa persona e seus insights acionáveis.
Contexto da análise:
{combined_context}
"""

try:
    response = model.generate_content(persona_prompt)
    return response.text.strip()
except Exception as e:
    st.error(f"Erro ao gerar insights para personas sintéticas: {e}")
    return "Não foi possível gerar insights para personas sintéticas."
@st.cache_data(show_spinner=True)
def generate_ice_score_tests(analysis_results):
sentiment = analysis_results.get('sentiment', {})
topics = analysis_results.get('topics', [])
term_clusters = analysis_results.get('term_clusters', {})
topic_relations = analysis_results.get('topic_relations', [])

# Formata os dados JSON ANTES de inseri-los na f-string
sentiment_json = json.dumps(sentiment, ensure_ascii=False)
topics_json = json.dumps(topics, ensure_ascii=False)
term_clusters_json = json.dumps(term_clusters, ensure_ascii=False)
topic_relations_json = json.dumps(topic_relations, ensure_ascii=False)

prompt = f"""
Com base na análise de social listening fornecida, atue como um Growth Hacker experiente.
Sugira EXATAMENTE 10 testes de Growth priorizados usando a metodologia ICE Score (Impacto, Confiança, Facilidade).
Para cada teste, identifique UMA ÚNICA VARIÁVEL de alavancagem principal entre: "Canal", "Segmentação", "Formato", "Criativo" ou "Copy/Argumento".

Apresente os resultados em formato JSON, como uma lista de objetos. Não inclua nenhum texto adicional antes ou depois do JSON.

A metodologia ICE Score é:

Impacto (Impact): Escala de 1 a 10 (10 = alto impacto potencial)
Confiança (1-10): Escala de 1 a 10 (10 = alta confiança no sucesso)
Facilidade (1-10): Escala de 1 a 10 (10 = muito fácil de implementar, baixo custo, pouco tempo)
ICE Score = (Impacto + Confiança + Facilidade) / 3 (arredondar para duas casas decimais)
A lista de testes deve ser ordenada do maior para o menor ICE Score.

Informações da análise para basear os testes:

Sentimento Geral: {sentiment_json}
Temas Mais Citados: {topics_json}
Agrupamento de Termos: {term_clusters_json}
Relação entre Temas: {topic_relations_json}
Exemplo da estrutura JSON de um item da lista:

JSON

[
  {{
    "Ordem": 1,
    "Nome do Teste": "Teste de Manchete de Anúncio",
    "Descrição do Teste": "Testar diferentes manchetes para anúncios no Facebook Ads para aumentar o CTR, focando no tema 'Benefícios Previdenciários'.",
    "Variável de Alavancagem": "Copy/Argumento",
    "Impacto (1-10)": 9,
    "Confiança (1-10)": 8,
    "Facilidade (1-10)": 7,
    "ICE Score": 8.00
  }}
]
"""

try:
    response = model.generate_content(prompt)
    response_text = response.text.strip()

    if response_text.startswith("```json"):
        response_text = response_text[len("```json"):].strip()
    if response_text.endswith("```"):
        response_text = response_text[:-len("```")].strip()

    data = json.loads(response_text)
    return data
except json.JSONDecodeError as e:
    st.error(f"Erro ao decodificar JSON da resposta do Gemini para ICE Score: {e}")
    st.code(f"Resposta bruta do Gemini (ICE Score): {response_text}")
    return None
except Exception as e:
    st.error(f"Erro inesperado ao gerar testes de Growth com ICE Score: {e}")
    return None
--- Funções de Visualização ---
def plot_sentiment(sentiment_data):
if not sentiment_data or sum(sentiment_data.values()) == 0:
st.warning("Dados de sentimento insuficientes para plotar.")
return

labels = list(sentiment_data.keys())
sizes = [sentiment_data[key] for key in labels]
colors = ['#4CAF50', '#FFC107', '#F44336', '#9E9E9E'] # Positivo, Neutro, Negativo, Sem Sentimento

# Criar um "explode" para destacar a maior fatia, se houver
explode = [0.0 if not sizes else 0.05 if size == max(sizes) else 0.0 for size in sizes]


fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'color': 'black'})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Análise de Sentimento Geral', fontsize=16, pad=20)
st.pyplot(fig1)
def plot_topics_sentiment(topics_data):
if not topics_data:
st.warning("Dados de temas insuficientes para plotar.")
return

df_topics = pd.DataFrame(topics_data)
df_topics_melted = df_topics.melt(id_vars='name', var_name='sentiment_type', value_name='count')

# Excluir 'no_sentiment_detected' da visualização de temas para focar nos sentimentos explícitos
df_topics_melted = df_topics_melted[df_topics_melted['sentiment_type'] != 'no_sentiment_detected']

if df_topics_melted.empty:
    st.warning("Dados de temas com sentimentos detectados insuficientes para plotar.")
    return

# Definir uma paleta de cores para os sentimentos
sentiment_colors = {
    'positive': '#4CAF50',
    'neutral': '#FFC107',
    'negative': '#F44336'
}

fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='name', y='count', hue='sentiment_type', data=df_topics_melted, palette=sentiment_colors, ax=ax)
plt.title('Sentimento por Tema', fontsize=16)
plt.xlabel('Tema', fontsize=12)
plt.ylabel('Contagem de Comentários', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Sentimento', title_fontsize='12', fontsize='10')
plt.tight_layout()
st.pyplot(fig)
def plot_term_clusters(term_clusters_data):
if not term_clusters_data:
st.warning("Dados de agrupamento de termos insuficientes para plotar.")
return

df_terms = pd.DataFrame(list(term_clusters_data.items()), columns=['Termo', 'Frequência'])
df_terms = df_terms.sort_values(by='Frequência', ascending=False)

if df_terms.empty:
    st.warning("Nenhum termo para plotar.")
    return

# Limitar aos top N termos para visualização clara
top_n = min(20, len(df_terms)) # Limita a 20 ou menos se tiver menos
df_terms = df_terms.head(top_n)

fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.5))) # Tamanho ajustável
sns.barplot(x='Frequência', y='Termo', data=df_terms, palette='viridis', ax=ax)
plt.title('Top Termos Frequentes', fontsize=16)
plt.xlabel('Frequência', fontsize=12)
plt.ylabel('Termo', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
st.pyplot(fig)
def plot_topic_relations(topic_relations_data, topics_data):
if not topic_relations_data:
st.warning("Dados de relação entre temas insuficientes para plotar o grafo.")
return

G = nx.Graph()

# Adiciona nós (temas)
for topic in topics_data:
    G.add_node(topic['name'])

# Adiciona arestas (relações)
for rel in topic_relations_data:
    # Verifica se os temas existem como nós antes de adicionar a aresta
    if rel['source'] in G and rel['target'] in G:
        G.add_edge(rel['source'], rel['target'], description=rel['description'])
    else:
        st.warning(f"Tema '{rel['source']}' ou '{rel['target']}' da relação não encontrado na lista de temas. Pulando aresta.")

if not G.edges():
    st.warning("Não há arestas suficientes para plotar um grafo de rede significativo. Verifique as relações entre temas.")
    return

fig, ax = plt.subplots(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42) # Layout baseado em força para melhor visualização

# Cores baseadas nos clusters identificados
# Gerar cores distintas para cada cluster
num_components = nx.number_connected_components(G)
colors_list = sns.color_palette("tab10", n_colors=max(num_components, 1)) # Garante pelo menos uma cor

# Atribuir cores aos nós com base nos componentes conectados
node_colors = []
for i, component in enumerate(nx.connected_components(G)):
    for node in component:
        node_colors.append(colors_list[i % len(colors_list)])

# Desenhar os nós
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9, ax=ax)

# Desenhar as arestas
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray', ax=ax)

# Desenhar os rótulos dos nós
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

# Adicionar rótulos das arestas (descrição da relação) - Opcional, pode poluir o grafo
# edge_labels = nx.get_edge_attributes(G, 'description')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

ax.set_title("Relação entre Temas", fontsize=18, pad=20)
ax.axis('off')
plt.tight_layout()
st.pyplot(fig)
--- Aplicação Streamlit ---
st.set_page_config(layout="wide", page_title="Análise de Social Listening com Gemini")

st.title("🗣️ Análise de Social Listening com Gemini")
st.markdown("---")

st.sidebar.header("Fonte dos Dados")
data_source_option = st.sidebar.radio(
"Escolha a fonte dos comentários:",
("Upload de Arquivo (CSV, Excel, Word)", "URL de Vídeo do YouTube")
)

all_comments_list = []

if data_source_option == "Upload de Arquivo (CSV, Excel, Word)":
uploaded_file = st.sidebar.file_uploader(
"Faça o upload do seu arquivo de comentários (.csv, .xls, .xlsx, .doc, .docx)",
type=["csv", "xls", "xlsx", "doc", "docx"]
)
if uploaded_file is not None:
file_extension = os.path.splitext(uploaded_file.name)[1].lower()
file_contents = uploaded_file.read()
all_comments_list = extract_text_from_file(file_contents, file_extension)
if all_comments_list:
st.sidebar.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso. {len(all_comments_list)} comentários extraídos.")
st.sidebar.write("Amostra dos comentários:")
for i, comment in enumerate(all_comments_list[:5]):
st.sidebar.text(f"- {comment[:70]}...") # Mostra os primeiros 70 caracteres
else:
st.sidebar.warning("Nenhum comentário válido foi extraído do arquivo. Verifique o formato ou o conteúdo da coluna 'comentario'.")

elif data_source_option == "URL de Vídeo do YouTube":
youtube_url_input = st.sidebar.text_input("Insira a URL do vídeo do YouTube:")
if youtube_url_input:
all_comments_list = download_youtube_comments(youtube_url_input)
if all_comments_list:
st.sidebar.success(f"Comentários baixados com sucesso. Total: {len(all_comments_list)}.")
st.sidebar.write("Amostra dos comentários:")
for i, comment in enumerate(all_comments_list[:5]):
st.sidebar.text(f"- {comment[:70]}...")
else:
st.sidebar.warning("Não foi possível baixar comentários. Verifique a URL ou as configurações de privacidade do vídeo.")

st.markdown("---")

if all_comments_list:
st.header("Comentários Coletados")
st.info(f"Total de {len(all_comments_list)} comentários coletados e prontos para análise.")

# Concatena todos os comentários em uma única string para análise do Gemini
# Limitar o tamanho da string para evitar sobrecarga da API do Gemini
MAX_TEXT_LENGTH_FOR_GEMINI = 100000 # ~100k caracteres (ajustável)
combined_comments = "\n".join(all_comments_list)
if len(combined_comments) > MAX_TEXT_LENGTH_FOR_GEMINI:
    st.warning(f"O número de caracteres total dos comentários ({len(combined_comments)}) excede o limite recomendado para a API ({MAX_TEXT_LENGTH_FOR_GEMINI})."
               " A análise será realizada com uma amostra truncada dos comentários. Para análises mais profundas, considere segmentar o input.")
    combined_comments = combined_comments[:MAX_TEXT_LENGTH_FOR_GEMINI] + "..."

st.write("Iniciando análise com IA (Gemini)...")

# Armazenar os resultados da análise no estado da sessão para evitar re-execução
if 'analysis_results' not in st.session_state or st.session_state.get('last_combined_comments') != combined_comments:
    st.session_state.analysis_results = analyze_text_with_gemini(combined_comments)
    st.session_state.last_combined_comments = combined_comments
    st.session_state.original_text_sample = combined_comments # Armazenar a amostra para a persona

analysis_results = st.session_state.analysis_results

if analysis_results:
    st.success("Análise concluída!")

    # Abas para organizar os resultados
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Sentimento", "💡 Temas", "☁️ Termos-Chave",
        "🔗 Relações entre Temas", "📝 Análise Qualitativa", "🚀 Testes de Growth (ICE Score)"
    ])

    with tab1:
        st.header("Análise de Sentimento Geral")
        plot_sentiment(analysis_results.get('sentiment', {}))
        st.json(analysis_results.get('sentiment', {}))

    with tab2:
        st.header("Temas Mais Citados")
        plot_topics_sentiment(analysis_results.get('topics', []))
        st.json(analysis_results.get('topics', []))

    with tab3:
        st.header("Agrupamento de Termos/Nuvem de Palavras")
        plot_term_clusters(analysis_results.get('term_clusters', {}))
        st.json(analysis_results.get('term_clusters', {}))

    with tab4:
        st.header("Relação entre Temas")
        # Garanta que topic_relations tenha descrições para o grafo
        plot_topic_relations(analysis_results.get('topic_relations', []), analysis_results.get('topics', []))
        st.json(analysis_results.get('topic_relations', []))

    with tab5:
        st.header("Análise Qualitativa Detalhada")
        qualitative_analysis = generate_qualitative_analysis(analysis_results)
        st.write(qualitative_analysis)

        st.header("Insights de Persona Sintética")
        persona_insights = generate_persona_insights(analysis_results, st.session_state.original_text_sample)
        st.write(persona_insights)

    with tab6:
        st.header("Sugestões de Testes de Growth (ICE Score)")
        ice_tests = generate_ice_score_tests(analysis_results)
        if ice_tests:
            df_ice = pd.DataFrame(ice_tests)
            # Recalcular ICE Score para garantir consistência
            df_ice['ICE Score'] = (df_ice['Impacto (1-10)'] + df_ice['Confiança (1-10)'] + df_ice['Facilidade (1-10)']) / 3
            df_ice['ICE Score'] = df_ice['ICE Score'].round(2)
            df_ice = df_ice.sort_values(by='ICE Score', ascending=False)
            df_ice.index = range(1, len(df_ice) + 1) # Reinicia o índice para 1
            df_ice.index.name = "Ordem"
            st.dataframe(df_ice)
        else:
            st.warning("Não foi possível gerar sugestões de testes de Growth.")
else:
    st.error("Não foi possível obter resultados da análise. Verifique os logs para mais detalhes.")
else:
st.info("Por favor, carregue um arquivo ou insira uma URL de vídeo do YouTube para iniciar a análise.")

st.markdown("---")
st.markdown("Desenvolvido com ❤️ e IA por [Seu Nome/Empresa]") # Opcional: Adicione seu nome/empresa

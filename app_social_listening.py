import streamlit as st
import pandas as pd
import json
import io
import os
from docx import Document
from youtube_comment_downloader import YoutubeCommentDownloader
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import random

# --- Configuração da API Gemini ---
import google.generativeai as genai

gemini_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    st.error("Chave da API do Google Gemini não encontrada. Configure no painel de Secrets do Streamlit Cloud ou como variável de ambiente.")
    st.stop()

genai.configure(api_key=gemini_api_key)
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# --- Paleta de Cores Lovable ---
CUSTOM_COLORS = {
    'primary': '#fe1874',
    'secondary': '#1f2329',
    'background': '#f3f3f3',
    'positive_light_pink': '#ff99b0'
}

# --- Funções de Extração ---
@st.cache_data(show_spinner=False)
def extract_text_from_file(file_contents, file_extension):
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
            st.warning(f"Formato de arquivo não suportado: {file_extension}.")
            return []
    except Exception as e:
        st.error(f"Erro ao extrair texto do arquivo: {e}")
        return []
    return text_content_list

@st.cache_data(show_spinner=False)
def download_youtube_comments(youtube_url):
    MAX_COMMENTS_LIMIT = 2000
    try:
        downloader = YoutubeCommentDownloader()
        match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', youtube_url)
        video_id = match.group(1) if match else None
        if not video_id:
            st.error("URL inválida.")
            return []
        with st.spinner("Baixando comentários..."):
            all_comments = []
            for i, comment in enumerate(downloader.get_comments(video_id)):
                all_comments.append(comment['text'])
                if i + 1 >= MAX_COMMENTS_LIMIT:
                    break
        return all_comments
    except Exception as e:
        st.error(f"Erro no download: {e}")
        return []

# --- Função Robust JSON Cleaner ---
def clean_json_response(response_text):
    cleaned = re.sub(r"^```.*?\n|\n```$", "", response_text.strip(), flags=re.DOTALL).strip()
    cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
    cleaned = re.sub(r'(\{|,)\s*(\w+)\s*:', r'\1 "\2":', cleaned)
    cleaned = re.sub(r'([}\]"])\s*([{\["])', r'\1,\2', cleaned)
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
    return cleaned

# --- Função Análise com Gemini ---
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
    2.  **Temas Mais Citados:** Uma lista dos 5 a 10 temas principais discutidos, com contagem de sentimentos.
    3.  **Agrupamento de Termos/Nuvem de Palavras:** Lista dos 10 a 20 termos-chave mais frequentes (sem palavras de parada), com contagem.
    4.  **Relação entre Temas:** De 3 a 5 pares de temas frequentemente citados juntos, e descrição da relação.

    Estrutura:
    {{
      "sentiment": {{
        "positive": float,
        "neutral": float,
        "negative": float,
        "no_sentiment_detected": float
      }},
      "topics": [
        {{
          "name": "Tema",
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
          "description": "Descrição"
        }}
      ]
    }}
    Texto para análise:
    "{text_to_analyze}"
    """

    try:
        response = model.generate_content(prompt)
        response_text = clean_json_response(response.text)
        data = json.loads(response_text)
        return data
    except Exception as e:
        st.error(f"Erro ao analisar/parsear JSON: {e}")
        st.code(response_text, language="json")
        return None

# --- Geração de Análises Qualitativas e Personas ---
def generate_qualitative_analysis(analysis_results, original_text_sample):
    sentiment = analysis_results.get('sentiment', {})
    topics = analysis_results.get('topics', [])
    term_clusters = analysis_results.get('term_clusters', {})
    topic_relations = analysis_results.get('topic_relations', [])
    prompt = f"""
    Com base na análise de social listening dos comentários de redes sociais fornecidos, redija uma análise qualitativa abrangente em até 4 parágrafos, focando nos aprendizados e insights estratégicos. Considere:
    Sentimento Geral: {json.dumps(sentiment)}
    Temas Mais Citados: {json.dumps(topics, ensure_ascii=False)}
    Agrupamento de Termos: {json.dumps(term_clusters, ensure_ascii=False)}
    Relação entre Temas: {json.dumps(topic_relations, ensure_ascii=False)}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Erro na análise qualitativa: {e}")
        return ""

def generate_persona_insights(analysis_results, original_text_sample):
    sentiment = analysis_results.get('sentiment', {})
    topics = analysis_results.get('topics', [])
    term_clusters = analysis_results.get('term_clusters', {})
    original_text_display = original_text_sample[:1000] + "..." if len(original_text_sample) > 1000 else original_text_sample
    prompt = f"""
    Com base na análise de social listening fornecida, crie uma persona sintética: dores, interesses, tom de comunicação, oportunidades de engajamento, e um nome sugestivo para a persona. 
    Comentários originais (amostra): {original_text_display}
    Resultados: Sentimento: {json.dumps(sentiment)}, Temas: {json.dumps(topics, ensure_ascii=False)}, Termos: {json.dumps(term_clusters, ensure_ascii=False)}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Erro ao gerar persona: {e}")
        return ""

def generate_ice_score_tests(analysis_results):
    sentiment = analysis_results.get('sentiment', {})
    topics = analysis_results.get('topics', [])
    term_clusters = analysis_results.get('term_clusters', {})
    topic_relations = analysis_results.get('topic_relations', [])
    prompt = f"""
    Com base na análise de social listening fornecida, sugira EXATAMENTE 10 testes de Growth priorizados usando ICE Score (Impacto, Confiança, Facilidade). 
    Para cada teste, informe uma variável principal de alavancagem: "Canal", "Segmentação", "Formato", "Criativo" ou "Copy/Argumento". 
    Retorne apenas JSON, ordenado por ICE Score decrescente.
    """
    try:
        response = model.generate_content(prompt)
        response_text = clean_json_response(response.text)
        data = json.loads(response_text)
        return data
    except Exception as e:
        st.error(f"Erro ao gerar ICE Score: {e}")
        st.code(response_text, language="json")
        return None

# --- Funções de Plotagem ---
def plot_sentiment_chart(sentiment_data):
    if not sentiment_data or sum(sentiment_data.values()) == 0:
        st.warning("Dados de sentimento vazios.")
        return
    labels_order = ['positive', 'neutral', 'negative', 'no_sentiment_detected']
    display_labels = ['Positivo', 'Neutro', 'Negativo', 'Não Detectado']
    colors_for_pie = {
        'positive': CUSTOM_COLORS['positive_light_pink'],
        'neutral': CUSTOM_COLORS['secondary'],
        'negative': CUSTOM_COLORS['primary'],
        'no_sentiment_detected': '#CCCCCC'
    }
    sizes = [sentiment_data.get(label, 0.0) for label in labels_order]
    filtered_data = [(display_labels[i], sizes[i], colors_for_pie[labels_order[i]])
                     for i, size in enumerate(sizes) if size > 0]
    if not filtered_data:
        st.warning("Nenhum dado de sentimento para plotar.")
        return
    filtered_labels, filtered_sizes, filtered_colors = zip(*filtered_data)
    explode = [0.03] * len(filtered_labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(filtered_sizes, explode=explode, labels=filtered_labels,
                                      colors=filtered_colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    for autotext in autotexts:
        autotext.set_color(CUSTOM_COLORS['background'])
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_color(CUSTOM_COLORS['secondary'])
        text.set_fontsize(12)
    centre_circle = plt.Circle((0,0),0.70,fc=CUSTOM_COLORS['background'])
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    ax.set_title('1. Análise de Sentimento Geral', pad=20, color=CUSTOM_COLORS['secondary'])
    st.pyplot(fig)

def plot_topics_chart(topics_data):
    if not topics_data:
        st.warning("Dados de temas vazios.")
        return
    df_topics = pd.DataFrame(topics_data)
    if df_topics.empty:
        st.warning("Nenhum dado de temas para plotar.")
        return
    df_topics['positive'] = df_topics['positive'].fillna(0).astype(int)
    df_topics['neutral'] = df_topics['neutral'].fillna(0).astype(int)
    df_topics['negative'] = df_topics['negative'].fillna(0).astype(int)
    df_topics['Total'] = df_topics['positive'] + df_topics['neutral'] + df_topics['negative']
    df_topics = df_topics.sort_values('Total', ascending=True)
    bar_colors = [CUSTOM_COLORS['positive_light_pink'], CUSTOM_COLORS['secondary'], CUSTOM_COLORS['primary']]
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_topics) * 0.7)))
    df_topics[['positive', 'neutral', 'negative']].plot(
        kind='barh', stacked=True, color=bar_colors, ax=ax)
    ax.set_title('2. Temas Mais Citados por Sentimento', color=CUSTOM_COLORS['secondary'])
    ax.set_xlabel('Número de Comentários', color=CUSTOM_COLORS['secondary'])
    ax.set_ylabel('Tema', color=CUSTOM_COLORS['secondary'])
    ax.set_yticklabels(df_topics['name'], color=CUSTOM_COLORS['secondary'])
    ax.tick_params(axis='x', colors=CUSTOM_COLORS['secondary'])
    ax.tick_params(axis='y', colors=CUSTOM_COLORS['secondary'])
    ax.legend(['Positivo', 'Neutro', 'Negativo'], loc='lower right', frameon=False, labelcolor=CUSTOM_COLORS['secondary'])
    plt.tight_layout()
    st.pyplot(fig)

def plot_word_cloud(term_clusters_data):
    if not term_clusters_data:
        st.warning("Dados de termos vazios.")
        return
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        colors = [CUSTOM_COLORS['primary'], CUSTOM_COLORS['secondary']]
        return random.choice(colors)
    wordcloud = WordCloud(width=1000, height=600, background_color=CUSTOM_COLORS['background'],
                          color_func=color_func, min_font_size=10, max_words=200,
                          contour_width=0, collocations=False).generate_from_frequencies(term_clusters_data)
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('3. Agrupamento de Termos (Nuvem de Palavras)', pad=20, color=CUSTOM_COLORS['secondary'])
    st.pyplot(fig)

def plot_topic_relations_chart(topic_relations_data):
    if not topic_relations_data:
        st.warning("Dados de relações entre temas vazios.")
        return
    G = nx.Graph()
    for rel in topic_relations_data:
        source = rel.get('source')
        target = rel.get('target')
        description = rel.get('description')
        if source and target:
            G.add_edge(source, target, description=description)
    if not G.edges():
        st.warning("Nenhuma relação válida para o grafo de rede.")
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)
    node_colors = [CUSTOM_COLORS['primary'] for _ in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color=CUSTOM_COLORS['secondary'], alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color=CUSTOM_COLORS['secondary'], ax=ax)
    ax.set_title('4. Relação Entre Temas (Grafo de Rede)', pad=20, color=CUSTOM_COLORS['secondary'])
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="Análise de Social Listening com Gemini")
st.title("🗣️ Análise de Social Listening com Gemini (by Pedro)")

st.sidebar.header("Fonte dos Dados")
data_source_option = st.sidebar.radio(
    "Escolha a fonte dos comentários:",
    ("Colar Texto", "Upload de Arquivo (CSV, Excel, Word)", "URL de Vídeo do YouTube")
)

all_comments_list = []

if data_source_option == "Colar Texto":
    user_text = st.sidebar.text_area("Cole seus comentários ou feedbacks (um por linha):", height=200)
    if user_text:
        all_comments_list = [line.strip() for line in user_text.split('\n') if line.strip()]
elif data_source_option == "Upload de Arquivo (CSV, Excel, Word)":
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do arquivo (.csv, .xls, .xlsx, .doc, .docx)",
        type=["csv", "xls", "xlsx", "doc", "docx"]
    )
    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_contents = uploaded_file.read()
        all_comments_list = extract_text_from_file(file_contents, file_extension)
elif data_source_option == "URL de Vídeo do YouTube":
    youtube_url = st.sidebar.text_input("URL do vídeo do YouTube:")
    if youtube_url:
        all_comments_list = download_youtube_comments(youtube_url)

if all_comments_list:
    st.success(f"Total de comentários coletados: {len(all_comments_list)}")
    with st.expander("Ver primeiros comentários (amostra)", expanded=False):
        for i, comment in enumerate(all_comments_list[:10]):
            st.write(f"- {comment[:100]}{'...' if len(comment) > 100 else ''}")
    combined_comments = "\n".join(all_comments_list)
    if 'analysis_results' not in st.session_state or st.session_state.get('last_combined_comments') != combined_comments:
        with st.spinner("Analisando comentários com Gemini..."):
            st.session_state.analysis_results = analyze_text_with_gemini(combined_comments)
        st.session_state.last_combined_comments = combined_comments
        st.session_state.original_text_sample = combined_comments

    analysis_results = st.session_state.analysis_results
    if analysis_results:
        st.header("🔎 Resultados da Análise de Social Listening")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊

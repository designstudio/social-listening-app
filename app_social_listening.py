import streamlit as st
import pandas as pd
import json
import io
from docx import Document
from youtube_comment_downloader import YoutubeCommentDownloader
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import google.generativeai as genai

# --- Configuração da API Gemini ---
gemini_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    st.error("Chave da API do Google Gemini não encontrada. Configure-a no painel de Secrets ou como variável de ambiente.")
    st.stop()

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Funções Auxiliares ---
def clean_json_response(response_text):
    cleaned = re.sub(r"^```.*?\n|\n```$", "", response_text.strip(), flags=re.DOTALL).strip()
    cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
    cleaned = re.sub(r'(\{|,)\s*(\w+)\s*:', r'\1 "\2":', cleaned)
    cleaned = re.sub(r'([}\]"])\s*([{\["])', r'\1,\2', cleaned)
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
    return cleaned

# --- Extração de Dados ---
@st.cache_data(show_spinner=False)
def extract_text_from_file(file_contents, file_extension):
    text_content_list = []
    try:
        if file_extension == '.csv':
            df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
            text_content_list = df['comentario'].dropna().astype(str).tolist() if 'comentario' in df.columns else \
                [" ".join(row.dropna().astype(str).tolist()) for _, row in df.iterrows()]
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(io.BytesIO(file_contents))
            text_content_list = df['comentario'].dropna().astype(str).tolist() if 'comentario' in df.columns else \
                [" ".join(row.dropna().astype(str).tolist()) for _, row in df.iterrows()]
        elif file_extension in ['.doc', '.docx']:
            document = Document(io.BytesIO(file_contents))
            text_content_list = [para.text for para in document.paragraphs if para.text.strip()]
        else:
            st.warning(f"Formato de arquivo não suportado: {file_extension}.")
            return []
    except Exception as e:
        st.error(f"Erro ao extrair texto: {e}")
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

# --- Análise com Gemini ---
def analyze_text_with_gemini(text_to_analyze):
    if not text_to_analyze.strip():
        return {"sentiment": {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "no_sentiment_detected": 100.0},
                "topics": [], "term_clusters": {}, "topic_relations": []}

    prompt = f"""
Analise o seguinte texto e retorne APENAS o JSON válido, com todas as chaves e strings entre aspas duplas.
Sem blocos de código, sem texto antes ou depois, apenas o JSON.
Texto:
{text_to_analyze}
"""

    try:
        response = model.generate_content(prompt)
        response_text = clean_json_response(response.text)
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError as e:
        st.error(f"Erro ao converter JSON: {e}")
        st.warning("Verifique se o texto analisado está muito grande ou se o Gemini retornou algo fora do padrão.")
        st.code(response_text, language="json")
        return None
    except Exception as e:
        st.error(f"Erro inesperado: {e}")
        return None

def generate_qualitative_analysis(analysis_results):
    try:
        context = json.dumps(analysis_results, ensure_ascii=False)
        prompt = f"Com base neste JSON: {context}, escreva uma análise qualitativa com até 4 parágrafos."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Erro: {e}")
        return "Não foi possível gerar a análise."

def generate_persona_insights(analysis_results, original_text_sample):
    try:
        context = json.dumps(analysis_results, ensure_ascii=False)
        prompt = f"Com base neste JSON: {context} e no texto: {original_text_sample[:1000]}, descreva uma persona sintética."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Erro: {e}")
        return "Não foi possível gerar a persona."

def generate_ice_score_tests(analysis_results):
    try:
        context = json.dumps(analysis_results, ensure_ascii=False)
        prompt = f"Com base neste JSON: {context}, gere 10 testes de growth com ICE Score, no formato JSON."
        response = model.generate_content(prompt)
        response_text = clean_json_response(response.text)
        data = json.loads(response_text)
        return data
    except Exception as e:
        st.error(f"Erro no ICE Score: {e}")
        return None

# --- Visualizações ---
def plot_sentiment(sentiment_data):
    if not sentiment_data or sum(sentiment_data.values()) == 0:
        st.warning("Dados insuficientes para o gráfico.")
        return
    labels = list(sentiment_data.keys())
    sizes = [sentiment_data[key] for key in labels]
    colors = ['#4CAF50', '#FFC107', '#F44336', '#9E9E9E']
    explode = [0.05 if size == max(sizes) else 0 for size in sizes]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

def plot_topics_sentiment(topics_data):
    if not topics_data:
        st.warning("Dados insuficientes para o gráfico.")
        return
    df = pd.DataFrame(topics_data).melt(id_vars='name', var_name='sentiment', value_name='count')
    df = df[df['sentiment'] != 'no_sentiment_detected']
    colors = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#F44336'}
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='name', y='count', hue='sentiment', data=df, palette=colors, ax=ax)
    ax.set_title("Sentimento por Tema")
    ax.set_xlabel("Tema")
    ax.set_ylabel("Contagem")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def plot_term_clusters(term_clusters_data):
    if not term_clusters_data:
        st.warning("Dados insuficientes para o gráfico.")
        return
    df = pd.DataFrame(list(term_clusters_data.items()), columns=['Termo', 'Frequência']).sort_values(by='Frequência', ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Frequência', y='Termo', data=df, palette='viridis', ax=ax)
    ax.set_title("Top Termos Frequentes")
    st.pyplot(fig)

def plot_topic_relations(topic_relations_data, topics_data):
    if not topic_relations_data:
        st.warning("Dados insuficientes para o grafo.")
        return
    G = nx.Graph()
    for topic in topics_data:
        G.add_node(topic['name'])
    for rel in topic_relations_data:
        if rel['source'] in G and rel['target'] in G:
            G.add_edge(rel['source'], rel['target'])
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000, font_size=10, font_weight='bold', ax=ax)
    ax.set_title("Relação entre Temas")
    st.pyplot(fig)

# --- App Streamlit ---
st.set_page_config(layout="wide", page_title="Análise de Social Listening com Gemini")
st.title("🗣️ Análise de Social Listening com Gemini")
st.markdown("---")

st.sidebar.header("Fonte dos Dados")
data_source_option = st.sidebar.radio("Escolha a fonte:", ("Upload de Arquivo (CSV, Excel, Word)", "URL de Vídeo do YouTube"))
all_comments_list = []

if data_source_option == "Upload de Arquivo (CSV, Excel, Word)":
    uploaded_file = st.sidebar.file_uploader("Faça upload:", type=["csv", "xls", "xlsx", "doc", "docx"])
    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_contents = uploaded_file.read()
        all_comments_list = extract_text_from_file(file_contents, file_extension)
        if all_comments_list:
            st.sidebar.success(f"Arquivo carregado: {len(all_comments_list)} comentários.")
elif data_source_option == "URL de Vídeo do YouTube":
    youtube_url = st.sidebar.text_input("URL do YouTube:")
    if youtube_url:
        all_comments_list = download_youtube_comments(youtube_url)
        if all_comments_list:
            st.sidebar.success(f"Comentários baixados: {len(all_comments_list)}.")

if all_comments_list:
    st.header("Comentários Coletados")
    combined_comments = "\n".join(all_comments_list)
    if 'analysis_results' not in st.session_state or st.session_state.get('last_combined_comments') != combined_comments:
        st.session_state.analysis_results = analyze_text_with_gemini(combined_comments)
        st.session_state.last_combined_comments = combined_comments
        st.session_state.original_text_sample = combined_comments

    analysis_results = st.session_state.analysis_results
    if analysis_results:
        st.success("Análise concluída!")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Sentimento", "💡 Temas", "☁️ Termos", "🔗 Relações", "📝 Análise Qualitativa", "🚀 Growth Tests"])
        with tab1:
            plot_sentiment(analysis_results.get('sentiment', {}))
        with tab2:
            plot_topics_sentiment(analysis_results.get('topics', []))
        with tab3:
            plot_term_clusters(analysis_results.get('term_clusters', {}))
        with tab4:
            plot_topic_relations(analysis_results.get('topic_relations', []), analysis_results.get('topics', []))
        with tab5:
            st.write(generate_qualitative_analysis(analysis_results))
            st.write(generate_persona_insights(analysis_results, st.session_state.original_text_sample))
        with tab6:
            tests = generate_ice_score_tests(analysis_results)
            if tests:
                df = pd.DataFrame(tests)
                df['ICE Score'] = ((df['Impacto (1-10)'] + df['Confiança (1-10)'] + df['Facilidade (1-10)']) / 3).round(2)
                st.dataframe(df)
            else:
                st.warning("Não foi possível gerar os testes.")

st.markdown("---")
st.markdown("Desenvolvido com ❤️ e IA por [Seu Nome/Empresa]")

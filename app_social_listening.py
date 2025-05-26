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

# ---- CONFIG GEMINI API ----
import google.generativeai as genai

gemini_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    st.error("Chave da API do Google Gemini não encontrada. Configure no painel de Secrets do Streamlit Cloud ou como variável de ambiente.")
    st.stop()

genai.configure(api_key=gemini_api_key)
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)

# ---- PALETA ----
CUSTOM_COLORS = {
    'primary': '#fe1874',
    'secondary': '#1f2329',
    'background': '#f3f3f3',
    'positive_light_pink': '#ff99b0'
}

# ---- EXTRAÇÃO ----
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

# ---- CLEAN JSON ----
def clean_json_response(response_text):
    cleaned = re.sub(r"^```.*?\n|\n```$", "", response_text.strip(), flags=re.DOTALL).strip()
    cleaned = cleaned.replace("'", '\"')
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
    return cleaned

# ---- GEMINI ANÁLISE ----
def analyze_text_with_gemini(text_to_analyze):
    if not text_to_analyze.strip():
        return {
            "sentiment": {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "no_sentiment_detected": 100.0},
            "topics": [],
            "term_clusters": {},
            "topic_relations": []
        }
    prompt = f"""
Analise o texto de comentários de redes sociais abaixo de forma estritamente objetiva, factual e consistente.
Extraia as informações solicitadas. Calcule as porcentagens e contagens EXATAS com base no total de comentários relevantes.
Retorne apenas o JSON, sem texto antes ou depois.
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
    "termo1": int
  }},
  "topic_relations": [
    {{
      "source": "Tema A",
      "target": "Tema B",
      "description": "Descrição"
    }}
  ]
}}
Sobre o agrupamento de termos ("term_clusters"): 
Retorne de 15 a 30 termos distintos e relevantes, com frequência real e variada (maior frequência = mais mencionado), 
evite empates desnecessários e garanta variedade temática.
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
        st.code(response_text if 'response_text' in locals() else '', language="json")
        return None

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
Retorne um array JSON PURO (começando em [ e terminando em ]), sem texto antes ou depois, sem comentários, sem vírgulas extras. Exemplo:
[
  {{
    "Ordem": 1,
    "Nome do Teste": "Teste de Criativo para Instagram",
    "Descrição do Teste": "Testar diferentes imagens no Instagram Ads...",
    "Variável de Alavancagem": "Criativo",
    "Impacto (1-10)": 9,
    "Confiança (1-10)": 8,
    "Facilidade (1-10)": 7,
    "ICE Score": 8.00
  }}
]
    """
    try:
        response = model.generate_content(prompt)
        response_text = clean_json_response(response.text)
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            # Tenta corrigir pequenas falhas comuns
            response_text_fixed = response_text.strip()
            if not response_text_fixed.endswith(']'):
                response_text_fixed += ']'
            response_text_fixed = response_text_fixed.replace('\n', '')
            response_text_fixed = re.sub(r',\s*([\]}])', r'\1', response_text_fixed)
            try:
                data = json.loads(response_text_fixed)
            except Exception as e2:
                st.error(f"Erro ao converter JSON do Gemini: {e2}")
                st.code(response_text, language="json")
                return None
        return data
    except Exception as e:
        st.error(f"Erro ao gerar ICE Score: {e}")
        st.code(response_text if 'response_text' in locals() else '', language="json")
        return None

# ---- VISUALIZAÇÃO ----
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

    # Seleciona os top 30 termos (ou mínimo 15) para a wordcloud, se possível
    top_n = max(15, min(30, len(term_clusters_data)))
    sorted_terms = dict(sorted(term_clusters_data.items(), key=lambda item: item[1], reverse=True)[:top_n])

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        cores = [CUSTOM_COLORS['primary'], CUSTOM_COLORS['secondary']]
        return random.choice(cores)

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color=CUSTOM_COLORS['background'],
        color_func=color_func,
        min_font_size=28,
        max_font_size=120,
        max_words=top_n,
        prefer_horizontal=0.95,
        scale=2,
        collocations=False,
        font_path=None
    ).generate_from_frequencies(sorted_terms)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.title('3. Agrupamento de Termos (Nuvem de Palavras)', fontsize=22, pad=18, color=CUSTOM_COLORS['secondary'])
    plt.tight_layout()
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

# ---- INTERFACE ----
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
    youtube_url_input = st.sidebar.text_input("Insira a URL do vídeo do YouTube:")
    if youtube_url_input:
        all_comments_list = download_youtube_comments(youtube_url_input)

st.markdown("---")

if all_comments_list:
    st.info(f"Total de {len(all_comments_list)} comentários coletados e prontos para análise.")
    MAX_TEXT_LENGTH_FOR_GEMINI = 100000
    combined_comments = "\n".join(all_comments_list)
    if len(combined_comments) > MAX_TEXT_LENGTH_FOR_GEMINI:
        st.warning(f"O número de caracteres total dos comentários ({len(combined_comments)}) excede o limite recomendado ({MAX_TEXT_LENGTH_FOR_GEMINI}). Usando amostra truncada.")
        combined_comments = combined_comments[:MAX_TEXT_LENGTH_FOR_GEMINI] + "..."

    if 'analysis_results' not in st.session_state or st.session_state.get('last_combined_comments') != combined_comments:
        st.session_state.analysis_results = analyze_text_with_gemini(combined_comments)
        st.session_state.last_combined_comments = combined_comments
        st.session_state.original_text_sample = combined_comments

    analysis_results = st.session_state.analysis_results

    if analysis_results:
        st.success("Análise concluída!")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Sentimento", "💡 Temas", "☁️ Termos-Chave",
            "🔗 Relações entre Temas", "📝 Análise Qualitativa", "🚀 Testes de Growth (ICE Score)"
        ])

        with tab1:
            st.header("Análise de Sentimento Geral")
            plot_sentiment_chart(analysis_results.get('sentiment', {}))
            st.json(analysis_results.get('sentiment', {}))

        with tab2:
            st.header("Temas Mais Citados")
            plot_topics_chart(analysis_results.get('topics', []))
            st.json(analysis_results.get('topics', []))

        with tab3:
            st.header("Agrupamento de Termos/Nuvem de Palavras")
            plot_word_cloud(analysis_results.get('term_clusters', {}))
            st.json(analysis_results.get('term_clusters', {}))

        with tab4:
            st.header("Relação entre Temas")
            plot_topic_relations_chart(analysis_results.get('topic_relations', []))
            st.json(analysis_results.get('topic_relations', []))

        with tab5:
            st.header("Análise Qualitativa Detalhada")
            qualitative_analysis = generate_qualitative_analysis(analysis_results, st.session_state.original_text_sample)
            if qualitative_analysis:
                st.write(qualitative_analysis)
            else:
                st.warning("A análise qualitativa não pôde ser gerada.")

            st.header("Insights de Persona Sintética")
            persona_insights = generate_persona_insights(analysis_results, st.session_state.original_text_sample)
            if persona_insights:
                st.write(persona_insights)
            else:
                st.warning("Não foi possível gerar insights de persona.")

        with tab6:
            st.header("Sugestões de Testes de Growth (ICE Score)")
            ice_tests = generate_ice_score_tests(analysis_results)
            if ice_tests and isinstance(ice_tests, list) and len(ice_tests) > 0:
                df_ice = pd.DataFrame(ice_tests)
                if "ICE Score" not in df_ice.columns:
                    for col in df_ice.columns:
                        if col.lower().replace(" ", "") in ["icescore", "ice", "ice_score"]:
                            df_ice.rename(columns={col: "ICE Score"}, inplace=True)
                        if col.lower() == "ordem":
                            df_ice.rename(columns={col: "Ordem"}, inplace=True)
                if "ICE Score" not in df_ice.columns and {'Impacto (1-10)', 'Confiança (1-10)', 'Facilidade (1-10)'}.issubset(df_ice.columns):
                    df_ice['ICE Score'] = (
                        df_ice['Impacto (1-10)'] + df_ice['Confiança (1-10)'] + df_ice['Facilidade (1-10)']) / 3
                    df_ice['ICE Score'] = df_ice['ICE Score'].round(2)
                if "ICE Score" in df_ice.columns:
                    df_ice = df_ice.sort_values(by='ICE Score', ascending=False)
                df_ice.index = range(1, len(df_ice) + 1)
                df_ice.index.name = "Ordem"
                st.dataframe(df_ice)
            else:
                st.warning("Não foi possível gerar sugestões de testes de Growth. Veja a resposta bruta abaixo para debugging:")
                st.code(json.dumps(ice_tests, indent=2, ensure_ascii=False), language="json")

    else:
        st.error("Não foi possível obter resultados da análise.")
else:
    st.info("Por favor, carregue um arquivo, cole comentários ou insira uma URL para iniciar a análise.")

st.markdown("---")
st.markdown("Desenvolvido com ❤️ e IA por Pedro Costa")  # Personalize como quiser

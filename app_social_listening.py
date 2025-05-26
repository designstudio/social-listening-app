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
from wordcloud import WordCloud
import os
import google.generativeai as genai

# --- CONFIGURAÇÃO GEMINI ---
gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
if not gemini_api_key:
    st.error("A chave da API do Google Gemini não foi encontrada. Configure-a no 'secrets.toml' ou como variável de ambiente GOOGLE_API_KEY.")
    st.stop()
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# --- EXTRAÇÃO DE DADOS ---
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
        video_id = None
        match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', youtube_url)
        if match:
            video_id = match.group(1)
        if not video_id:
            st.error(f"URL do YouTube inválida: {youtube_url}.")
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
                        break
            except Exception as e:
                st.error(f"Não foi possível baixar comentários: {e}")
                return []
            return all_comments
    except Exception as e:
        st.error(f"Erro geral ao baixar comentários: {e}.")
        return []

# --- ANÁLISE COM GEMINI ---
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
Analise o texto de comentários de redes sociais abaixo de forma estritamente objetiva, factual e consistente.
Extraia as informações solicitadas. Calcule as porcentagens e contagens EXATAS com base no total de comentários relevantes.
Forneça as seguintes informações em formato JSON, exatamente como a estrutura definida. Não inclua nenhum texto adicional antes ou depois do JSON.
1.  Sentimento Geral: A porcentagem de comentários classificados como 'Positivo', 'Neutro', 'Negativo' e 'Sem Sentimento Detectado'. As porcentagens devem somar 100%.
2.  Temas Mais Citados: Lista de 5 a 10 temas principais discutidos nos comentários, com contagem EXATA de comentários Positivos, Neutros e Negativos.
3.  Agrupamento de Termos/Nuvem de Palavras: Lista dos 10 a 20 termos ou palavras-chave mais frequentes. Para cada termo, forneça a frequência (contagem de ocorrências).
4.  Relação entre Temas: Liste 3 a 5 pares de temas que frequentemente aparecem juntos, indicando relação clara e lógica.
O JSON deve ter a seguinte estrutura:
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
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-len("```")].strip()
        data = json.loads(response_text)
        if 'no_sentiment_detected' not in data['sentiment']:
            total = data['sentiment'].get('positive', 0) + data['sentiment'].get('neutral', 0) + data['sentiment'].get('negative', 0)
            data['sentiment']['no_sentiment_detected'] = round(100.0 - total, 2)
        total_sum = sum(data['sentiment'].values())
        if total_sum != 100 and total_sum != 0:
            for key in data['sentiment']:
                data['sentiment'][key] = round(data['sentiment'][key] / total_sum * 100, 2)
        return data
    except json.JSONDecodeError as e:
        st.error(f"Erro ao decodificar JSON da resposta do Gemini: {e}")
        st.code(f"Resposta bruta do Gemini (Análise): {response_text}")
        return None
    except Exception as e:
        st.error(f"Erro inesperado ao analisar com Gemini: {e}")
        return None

@st.cache_data(show_spinner=True)
def generate_qualitative_analysis(analysis_results, original_text_sample):
    sentiment = analysis_results.get('sentiment', {})
    topics = analysis_results.get('topics', [])
    term_clusters = analysis_results.get('term_clusters', {})
    topic_relations = analysis_results.get('topic_relations', [])
    prompt = f"""
Com base na análise de social listening dos comentários, atue como um especialista em Marketing de Produto, Social Listening e Data Analysis. Redija uma análise qualitativa abrangente em até 4 parágrafos, focando nos aprendizados e insights estratégicos.
Considere as seguintes informações:
Sentimento Geral: {json.dumps(sentiment, ensure_ascii=False)}
Temas Mais Citados: {json.dumps(topics, ensure_ascii=False)}
Agrupamento de Termos: {json.dumps(term_clusters, ensure_ascii=False)}
Relação entre Temas: {json.dumps(topic_relations, ensure_ascii=False)}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Erro na análise qualitativa: {e}")
        return "Não foi possível gerar a análise qualitativa."

@st.cache_data(show_spinner=True)
def generate_ice_score_tests(analysis_results):
    sentiment = analysis_results.get('sentiment', {})
    topics = analysis_results.get('topics', [])
    term_clusters = analysis_results.get('term_clusters', {})
    topic_relations = analysis_results.get('topic_relations', [])
    prompt = f"""
Com base na análise de social listening fornecida, atue como um Growth Hacker experiente. Sugira EXATAMENTE 10 testes de Growth priorizados usando a metodologia ICE Score (Impacto, Confiança, Facilidade).
Para cada teste, identifique UMA ÚNICA VARIÁVEL de alavancagem principal entre: "Canal", "Segmentação", "Formato", "Criativo" ou "Copy/Argumento".
Apresente os resultados em formato JSON, como uma lista de objetos. Não inclua nenhum texto adicional antes ou depois do JSON.
A lista de testes deve ser ordenada do maior para o menor ICE Score.
Informações da análise:
Sentimento Geral: {json.dumps(sentiment, ensure_ascii=False)}
Temas Mais Citados: {json.dumps(topics, ensure_ascii=False)}
Agrupamento de Termos: {json.dumps(term_clusters, ensure_ascii=False)}
Relação entre Temas: {json.dumps(topic_relations, ensure_ascii=False)}
Exemplo:
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
        st.error(f"Erro ao gerar ICE Score: {e}")
        st.code(f"Resposta bruta do Gemini (ICE Score): {response_text}")
        return None
    except Exception as e:
        st.error(f"Erro inesperado ao gerar testes de Growth com ICE Score: {e}")
        return None

# --- VISUALIZAÇÃO ---

def plot_sentiment_chart(sentiment_data):
    labels_order = ['positive', 'neutral', 'negative', 'no_sentiment_detected']
    display_labels = ['Positivo', 'Neutro', 'Negativo', 'Não Detectado']
    colors_for_pie = {
        'positive': '#ff99b0',
        'neutral': '#1f2329',
        'negative': '#fe1874',
        'no_sentiment_detected': '#cccccc'
    }
    sizes = [sentiment_data.get(label, 0.0) for label in labels_order]
    filtered_data = [(display_labels[i], sizes[i], colors_for_pie[labels_order[i]])
                     for i, size in enumerate(sizes) if size > 0]
    if not filtered_data:
        st.warning("Dados insuficientes para gráfico de sentimento.")
        return
    filtered_labels, filtered_sizes, filtered_colors = zip(*filtered_data)
    explode = [0.03] * len(filtered_labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        filtered_sizes,
        explode=explode,
        labels=filtered_labels,
        colors=filtered_colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85
    )
    for autotext in autotexts:
        autotext.set_color('#f3f3f3')
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_color('#1f2329')
        text.set_fontsize(12)
    centre_circle = plt.Circle((0,0),0.70,fc='#f3f3f3')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    ax.set_title('1. Análise de Sentimento Geral', pad=20, color='#1f2329')
    st.pyplot(fig)

def plot_topics_chart(topics_data):
    if not topics_data:
        st.warning("Dados de temas insuficientes.")
        return
    df_topics = pd.DataFrame(topics_data)
    df_topics['positive'] = df_topics['positive'].fillna(0).astype(int)
    df_topics['neutral'] = df_topics['neutral'].fillna(0).astype(int)
    df_topics['negative'] = df_topics['negative'].fillna(0).astype(int)
    df_topics['Total'] = df_topics['positive'] + df_topics['neutral'] + df_topics['negative']
    df_topics = df_topics.sort_values('Total', ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_topics) * 0.7)))
    bar_colors = ['#ff99b0', '#1f2329', '#fe1874']
    df_topics[['positive', 'neutral', 'negative']].plot(
        kind='barh',
        stacked=True,
        color=bar_colors,
        ax=ax
    )
    ax.set_title('2. Temas Mais Citados por Sentimento', color='#1f2329')
    ax.set_xlabel('Número de Comentários', color='#1f2329')
    ax.set_ylabel('Tema', color='#1f2329')
    ax.set_yticklabels(df_topics['name'], color='#1f2329')
    ax.tick_params(axis='x', colors='#1f2329')
    ax.tick_params(axis='y', colors='#1f2329')
    ax.legend(['Positivo', 'Neutro', 'Negativo'], loc='lower right', frameon=False, labelcolor='#1f2329')
    plt.tight_layout()
    st.pyplot(fig)

def plot_word_cloud(term_clusters_data):
    if not term_clusters_data:
        st.warning("Dados de termos insuficientes para nuvem de palavras.")
        return
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        import random
        return '#fe1874' if random_state and random_state.randint(0, 2) == 0 else '#1f2329'
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color='#f3f3f3',
        color_func=color_func,
        min_font_size=16,
        max_words=60,
        prefer_horizontal=0.8,
        collocations=False
    ).generate_from_frequencies(term_clusters_data)
    fig = plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('3. Agrupamento de Termos (Nuvem de Palavras)', pad=20, fontsize=18)
    st.pyplot(fig)

def plot_topic_relations_chart(topic_relations_data):
    if not topic_relations_data:
        st.warning("Dados insuficientes para grafo de relação entre temas.")
        return
    G = nx.Graph()
    for rel in topic_relations_data:
        source = rel.get('source')
        target = rel.get('target')
        description = rel.get('description')
        if source and target:
            G.add_edge(source, target, description=description)
    if not G.edges():
        st.warning("Nenhuma relação válida encontrada para construir o grafo de rede.")
        return
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)
    node_colors = ['#fe1874' for _ in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='#1f2329', alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='#1f2329', ax=ax)
    ax.set_title('4. Relação Entre Temas (Grafo de Rede)', pad=20, color='#1f2329')
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# --- APP STREAMLIT ---
st.set_page_config(layout="wide", page_title="Social Listening Tool + AI")
st.title("🗣️ Social Listening Tool + AI")
st.markdown("---")

st.markdown("Carregue uma base de comentários (.csv, .xls, .xlsx, .doc, .docx), uma URL de vídeo do YouTube, ou cole comentários no campo abaixo. **Todos os gráficos e análises serão gerados automaticamente.**")

all_comments_list = []

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader(
        "Faça upload do arquivo de comentários (.csv, .xls, .xlsx, .doc, .docx):",
        type=["csv", "xls", "xlsx", "doc", "docx"],
        key="fileuploader"
    )
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_contents = uploaded_file.read()
        all_comments_list = extract_text_from_file(file_contents, file_extension)

with col2:
    youtube_url_input = st.text_input("Ou insira uma URL de vídeo do YouTube:")
    if youtube_url_input:
        yt_comments = download_youtube_comments(youtube_url_input.strip())
        if yt_comments:
            all_comments_list = yt_comments

manual_text = st.text_area("Ou cole comentários (um por linha):")
if manual_text and not all_comments_list:
    manual_comments = [l for l in manual_text.split("\n") if l.strip()]
    if manual_comments:
        all_comments_list = manual_comments
        st.success(f"{len(manual_comments)} comentários colados.")

if all_comments_list:
    st.success("Comentários carregados! Pronto para analisar.")
    text_to_analyze = "\n".join(all_comments_list)
    with st.spinner("Processando análise com Gemini..."):
        analysis_results = analyze_text_with_gemini(text_to_analyze)
    if analysis_results:
        tabs = st.tabs([
            "📊 Sentimento",
            "💡 Temas",
            "🔑 Termos-Chave",
            "🔗 Relações entre Temas",
            "📝 Análise Qualitativa",
            "🚀 Testes de Growth (ICE Score)"
        ])
        with tabs[0]:
            st.subheader("Sentimento Geral")
            plot_sentiment_chart(analysis_results.get('sentiment', {}))
        with tabs[1]:
            st.subheader("Temas mais Citados com Sentimento")
            plot_topics_chart(analysis_results.get('topics', []))
        with tabs[2]:
            st.subheader("Agrupamento de Termos/Nuvem de Palavras")
            plot_word_cloud(analysis_results.get('term_clusters', {}))
        with tabs[3]:
            st.subheader("Relação entre Temas (Grafo de Rede)")
            plot_topic_relations_chart(analysis_results.get('topic_relations', []))
        with tabs[4]:
            st.subheader("Análise Qualitativa")
            with st.spinner("Gerando análise qualitativa..."):
                qualitative = generate_qualitative_analysis(analysis_results, text_to_analyze)
                st.markdown(qualitative)
        with tabs[5]:
            st.subheader("Sugestões de Testes de Growth (ICE Score)")
            with st.spinner("Gerando sugestões de testes de growth..."):
                ice = generate_ice_score_tests(analysis_results)
                if ice:
                    df_ice = pd.DataFrame(ice)
                    cols = ["Ordem", "Nome do Teste", "Descrição do Teste", "Variável de Alavancagem", "Impacto (1-10)", "Confiança (1-10)", "Facilidade (1-10)", "ICE Score"]
                    df_ice = df_ice[[c for c in cols if c in df_ice.columns]]
                    df_ice = df_ice.sort_values(by="ICE Score", ascending=False)
                    st.dataframe(df_ice, hide_index=True, use_container_width=True)
                else:
                    st.warning("Não foi possível gerar sugestões de testes de growth.")
    else:
        st.error("Não foi possível gerar a análise com Gemini. Reveja os dados e tente novamente.")
else:
    st.info("Faça o upload de comentários, cole manualmente ou insira uma URL do YouTube para iniciar a análise.")



st.markdown("---")
st.markdown("Desenvolvido com Python, ❤️ e AI por Pedro Costa | Product Marketing & Martech Specialist")

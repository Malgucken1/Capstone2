import streamlit as st
import pymongo
import os
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import plotly.express as px

# =========================================================================
# 0. KONFIGURATION
# =========================================================================
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

MONGO_URI = st.secrets["mongo_uri"]
DATABASE_NAME = st.secrets["mongo_db"]
COLLECTION_NAME = st.secrets["mongo_collection"]

ATLAS_INDEX_NAME = "vector_index"
VECTOR_FIELD_NAME = "listing_embedding"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = "gpt-3.5-turbo"

# =========================================================================
# 1. DATENBANK & EMBEDDING INITIALISIEREN
# =========================================================================
@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGO_URI)
    return client[DATABASE_NAME][COLLECTION_NAME]

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# MongoDB Collection und Embedding-Modell laden
try:
    mongo_collection = get_mongo_collection()
    embedding_model = get_embedding_model()
    st.success("✅ Datenbank- und Modellverbindung erfolgreich hergestellt.")
except Exception as e:
    st.error(f"❌ Initialisierungsfehler: {e}")
    st.stop()

# =========================================================================
# 2. HELPER FUNKTION: RETRIEVAL AUS ATLAS
# =========================================================================
def retrieve_context(query_text, collection, embedding_model, limit=10):
    query_vector = embedding_model.encode(query_text).tolist()
    vector_search_pipeline = [
        {
            '$vectorSearch': {
                'index': ATLAS_INDEX_NAME,
                'path': VECTOR_FIELD_NAME,
                'queryVector': query_vector,
                'numCandidates': 100,
                'limit': limit,
            }
        },
        {
            '$project': {
                '_id': 0,
                'name': 1,
                'neighbourhood': 1,
                'room_type': 1,
                'price': 1,
                'score': {'$meta': 'vectorSearchScore'}
            }
        }
    ]
    results = list(collection.aggregate(vector_search_pipeline))
    context = ""
    for res in results:
        context += f"Listing Name: {res.get('name')}, Neighbourhood: {res.get('neighbourhood')}, Price: {res.get('price')}, Score: {res.get('score'):.4f}\n"
    return context, results

# =========================================================================
# 3. TOOL DEFINIEREN
# =========================================================================
def retrieval_tool(query: str) -> str:
    context, _ = retrieve_context(query, mongo_collection, embedding_model, limit=10)
    return context

tools = [
    Tool(
        name="Atlas-Retrieval",
        func=retrieval_tool,
        description="Führt eine semantische Suche in der MongoDB Atlas Vector-Datenbank durch, um relevante Airbnb-Listings zu finden."
    )
]

# =========================================================================
# 4. AGENT INITIALISIEREN (AGENTIC RAG)
# =========================================================================
AGENT_PROMPT = """
Du bist ein hilfreicher Agent, der dem Benutzer hilft, die besten Airbnb-Listings in Berlin zu finden.
Du kannst selbständig die Atlas-Retrieval-Funktion nutzen, um relevante Informationen abzurufen.
Vorgehensweise:
1. Analysiere die Benutzeranfrage.
2. Nutze das Tool 'Atlas-Retrieval', um relevante Listings zu finden.
3. Fasse die Ergebnisse in einer strukturierten, verständlichen Antwort zusammen.
4. Gib die Top-3 Listings mit Name, Viertel und Preis an.
"""

llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.0)
prompt = ChatPromptTemplate.from_template(AGENT_PROMPT)

# Agent und Executor erzeugen (neues API-Design)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# =========================================================================
# 5. STREAMLIT UI
# =========================================================================
st.set_page_config(page_title="Agentic RAG Airbnb Berlin 🏠", layout="wide")
st.title("Agentic RAG Chatbot - Airbnb Berlin (MongoDB Atlas)")

# Chatverlauf initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []

# Vorhandene Nachrichten anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Neue Benutzereingabe
if prompt_text := st.chat_input("Finde die beste Wohnung in Berlin..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Agent denkt nach und ruft Kontext ab..."):

            # Prüfen, ob der User einen Plot will
            if "plot" in prompt_text.lower() or "grafik" in prompt_text.lower():
                _, results = retrieve_context(prompt_text, mongo_collection, embedding_model, limit=50)
                df = pd.DataFrame(results)

                df_mitte = df[df['neighbourhood'].str.contains("Mitte", case=False, na=False)]

                if not df_mitte.empty:
                    fig = px.bar(
                        df_mitte,
                        x='name',
                        y='price',
                        title="Airbnb Listings in Berlin-Mitte",
                        labels={'price': 'Preis (€)', 'name': 'Listing Name'}
                    )
                    st.plotly_chart(fig)
                    response = "Hier ist ein Plot der Preise für Berlin-Mitte Listings 📊"
                else:
                    response = "Keine Listings für Berlin-Mitte gefunden, um einen Plot zu erstellen."
            else:
                # Standard Agentic RAG Antwort (neues API)
                try:
                    result = agent_executor.invoke({"input": prompt_text})
                    response = result["output"]
                except Exception as e:
                    response = f"❌ Fehler beim Agentenlauf: {e}"

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

import streamlit as st
import pymongo
import os
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.agents import Tool, initialize_agent, AgentType

# =========================================================================
# 0. KONFIGURATION
# =========================================================================
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

MONGO_URI = st.secrets["mongodb"]["uri"]
DATABASE_NAME = st.secrets["mongodb"]["database_name"]
COLLECTION_NAME = st.secrets["mongodb"]["collection_name"]

ATLAS_INDEX_NAME = "vector_index"
VECTOR_FIELD_NAME = "listing_embedding"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = "gpt-3.5-turbo"

# =========================================================================
# 1. FUNKTIONEN ZUM ZUGRIFF AUF DIE DATENBANK
# =========================================================================
@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGO_URI)
    return client[DATABASE_NAME][COLLECTION_NAME]

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def retrieve_context(query_text, collection, embedding_model, limit=5):
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
# 2. AGENTIC RAG-LOGIK
# =========================================================================
llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.0)

# Tool definieren: Retrieval aus Atlas
def retrieval_tool(query: str) -> str:
    context, _ = retrieve_context(query, mongo_collection, embedding_model, limit=5)
    return context

tools = [
    Tool(
        name="Atlas-Retrieval",
        func=retrieval_tool,
        description="Führt eine semantische Suche in der MongoDB Atlas Vector-Datenbank durch, um relevante Airbnb-Listings zu finden."
    )
]

# Prompt für Agent
AGENT_PROMPT = """
Du bist ein Agent, der dem Benutzer hilft, die besten Airbnb-Listings in Berlin zu finden.
Du kannst selbständig die Atlas-Retrieval-Funktion nutzen, um relevante Informationen abzurufen.
Gehe iterativ vor:
1. Plane, welche Informationen für die Benutzeranfrage nötig sind.
2. Rufe relevante Listings ab (ggf. mehrfach mit unterschiedlichen Suchanfragen).
3. Kombiniere die Ergebnisse und generiere eine Antwort.
4. Gib die Top-3 Listings mit Name, Viertel und Preis an.
"""

# Agent initialisieren (Agentic RAG)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # ermöglicht Tool-Aufrufe
    verbose=True,
    agent_kwargs={"prefix": AGENT_PROMPT}
)

# =========================================================================
# 3. STREAMLIT UI
# =========================================================================
st.set_page_config(page_title="Agentic RAG Airbnb Berlin 🏠", layout="wide")
st.title("Agentic RAG Chatbot - Airbnb Berlin (MongoDB Atlas)")

# Datenbank & Modell initialisieren
try:
    mongo_collection = get_mongo_collection()
    embedding_model = get_embedding_model()
    st.success("Datenbank- und Modellverbindung erfolgreich hergestellt.")
except Exception as e:
    st.error(f"❌ Initialisierungsfehler: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Zeige Chatverlauf
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Neue Benutzereingabe
if prompt := st.chat_input("Finde die beste Wohnung in Berlin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent denkt nach und ruft mehrfach Kontext ab..."):
            response = agent.run(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

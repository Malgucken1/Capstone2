import streamlit as st
import pymongo
import os
from sentence_transformers import SentenceTransformer
# Korrektur 3: Nutze den korrekten Import für Langchain Chat Models
from langchain_community.chat_models import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# =========================================================================
# 0. KONFIGURATION (Laden aus st.secrets)
# =========================================================================

# Korrektur 2: Lade den OpenAI API Key als Umgebungsvariable für LangChain
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"] 

# Korrektur 1: Lade Konfigurationen aus dem [mongodb]-Abschnitt der secrets.toml
MONGO_URI = st.secrets["mongodb"]["uri"]
DATABASE_NAME = st.secrets["mongodb"]["database_name"]
COLLECTION_NAME = st.secrets["mongodb"]["collection_name"]

# Diese Variablen bleiben als Konstanten im Code
ATLAS_INDEX_NAME = "vector_index" 
VECTOR_FIELD_NAME = "listing_embedding" 
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = "gpt-3.5-turbo" 

# =========================================================================
# 1. FUNKTIONEN ZUM ZUGRIFF AUF DIE DATENBANK
# =========================================================================

@st.cache_resource
def get_mongo_collection():
    """Initialisiert die MongoDB-Verbindung und gibt die Collection zurück."""
    # Der MongoDB Client findet den URI jetzt korrekt über die Variable MONGO_URI
    client = pymongo.MongoClient(MONGO_URI)
    return client[DATABASE_NAME][COLLECTION_NAME]

@st.cache_resource
def get_embedding_model():
    """Lädt und cached das Embedding-Modell."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def retrieve_context(query_text, collection, embedding_model, limit=5):
    """
    Führt die $vectorSearch in MongoDB Atlas durch, um relevanten Kontext abzurufen (R-Teil).
    """
    # 1. Anfrage vektorisieren
    query_vector = embedding_model.encode(query_text).tolist() 

    # 2. Vector Search Pipeline definieren
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
        # 3. Kontext extrahieren
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

    # 4. Abfrage ausführen und Ergebnisse formatieren
    results = list(collection.aggregate(vector_search_pipeline))
    
    # Kontext in einen String formatieren
    context = ""
    for res in results:
        context += f"Listing Name: {res.get('name')}, Neighbourhood: {res.get('neighbourhood')}, Price: {res.get('price')}, Score: {res.get('score'):.4f}\n"
        
    return context, results

# =========================================================================
# 2. CHATBOT-LOGIK (RAG-KETTE)
# =========================================================================

# LLM-Initialisierung
# Der API-Schlüssel wird automatisch über os.environ gefunden
llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.0)

# Prompt-Vorlage für den RAG-Chatbot (G-Teil)
RAG_PROMPT_TEMPLATE = """
Du bist ein hilfreicher Airbnb-Experte in Berlin.
Deine Aufgabe ist es, dem Benutzer basierend auf dem KONTEXT zu helfen.
Der Kontext besteht aus den semantisch ähnlichsten Listings in Berlin.

Antwort:
1. Nenne die Top-3-Listings, die der Anfrage des Benutzers am besten entsprechen.
2. Gib für jedes Listing den Namen, das Viertel (Neighbourhood) und den Preis an.
3. Wenn du die Information nicht zuverlässig aus dem Kontext ableiten kannst, sag höflich, dass du sie nicht hast.

KONTEXT:
{context}

FRAGE: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Langchain Expression Language (LEL) Kette
rag_chain = (
    {"context": lambda x: x["context"], "question": lambda x: x["question"]}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# =========================================================================
# 3. STREAMLIT UI
# =========================================================================

st.set_page_config(page_title="Atlas RAG Chatbot (Airbnb Berlin) 🏠", layout="wide")
st.title("Airbnb Berlin RAG Chatbot powered by MongoDB Atlas Vector Search")

# Datenbank und Modell initialisieren
try:
    mongo_collection = get_mongo_collection()
    embedding_model = get_embedding_model()
    st.success("Datenbank- und Modellverbindung erfolgreich hergestellt.")
except Exception as e:
    st.error(f"❌ Initialisierungsfehler: Konnte die Datenbank oder das Modell nicht laden. Fehler: {e}")
    st.stop()


# Initialisiere den Chat-Verlauf
if "messages" not in st.session_state:
    st.session_state.messages = []

# Zeige existierende Nachrichten an
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Verarbeite neue Benutzereingabe
if prompt := st.chat_input("Finde die beste Wohnung in Berlin..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Suche Listings in Atlas und generiere Antwort..."):
            
            # R-Teil: Retrieval
            context, results = retrieve_context(prompt, mongo_collection, embedding_model, limit=5)
            
            # G-Teil: Generation
            response = rag_chain.invoke({"context": context, "question": prompt})
            
            st.markdown(response)

        # Debugging / Transparenz
        with st.expander("Abgerufener Kontext (Debugging)"):
             st.markdown("---")
             st.markdown("Dies sind die 5 semantisch ähnlichsten Listings, die an das LLM gesendet wurden:")
             st.markdown(context)
             
        st.session_state.messages.append({"role": "assistant", "content": response})

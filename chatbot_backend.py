# chatbot_backend.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import re
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Load & scrape data from website
print("📡 Scraping latest data from JIMS site...")
docs = WebBaseLoader("https://ncjims.org/departments.html").load()

# 2. Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# 3. Translation functions
def translate_documents_to_hindi(docs: List):
    translated = []
    for d in docs:
        prompt = f"Translate this medical information into Hindi (Devanagari script):\n\n{d.page_content}"
        resp = llm.invoke(prompt).content
        translated.append(type(d)(page_content=resp, metadata=d.metadata.copy()))
    return translated

def translate_documents_to_hinglish(docs: List):
    translated = []
    for d in docs:
        prompt = f"""Convert this medical information to Hinglish (Latin script with Hindi words):
- Mix Hindi & English naturally
- Keep medical terms in English where appropriate

{d.page_content}"""
        resp = llm.invoke(prompt).content
        translated.append(type(d)(page_content=resp, metadata=d.metadata.copy()))
    return translated

# 4. Build multilingual corpus
english_docs  = docs
hindi_docs    = translate_documents_to_hindi(english_docs)
hinglish_docs = translate_documents_to_hinglish(english_docs)

for d in english_docs:   d.metadata["lang"] = "english"
for d in hindi_docs:     d.metadata["lang"] = "hindi"
for d in hinglish_docs:  d.metadata["lang"] = "hinglish"

all_docs = english_docs + hindi_docs + hinglish_docs

# 5. Create embeddings & vector store
print("📂 Building vector store...")
embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    collection_name="jims_docs_multilang",
    persist_directory="./chroma_db"
)

# 6. Language detection
def detect_language(text: str) -> str:
    if re.search(r'[\u0900-\u097F]', text):
        return "hindi"
    hinglish_words = {
        "hai", "hain", "nahi", "nahin", "kya","iske", "kyun", "kaise", "kahan",
        "mein", "main", "aur", "ya", "par", "pe", "ko", "ka", "ke", "ki","rha",
        "raha", "se", "me", "aap", "tum", "hum", "woh", "yeh", "aise", "waise",
        "kab", "kaun", "kitna", "kitni", "kitne", "lekin", "agar", "jab", "phir",
        "ab", "kyunki", "shayad", "bahut", "thoda", "zaroor", "chahiye", "kar",
        "bata", "puch", "mil", "namaste", "shukriya", "acha", "theek hai","inke"
        "koi baat nahi", "badiya", "mast", "chalo", "suno", "samjha", "samajh gaya",
        "samajh gayi", "bataye", "Janna","btao", "batao", "karo", "inki", "unka",
        "unki", "unke", "dikhaye", "rog", "ilaj"
    }
    words = set(re.findall(r'\b\w+\b', text.lower()))
    hinglish_count = len(words.intersection(hinglish_words))
    if hinglish_count >= 2 or (len(words) > 0 and hinglish_count / len(words) > 0.2):
        return "hinglish"
    return "english"

# 7. Base prompt (your existing rules)
base_prompt = """You are "JIMS-Guide", an assistant who answers questions using ONLY the data between === START-JIMS-DATA === and === END-JIMS-DATA ===.
 
Instructions:
1. Use chat history to maintain context and interpret short replies ("yes", "no", "ha", "nahi").
2. Greetings: Reply in the user’s language: “Hi, I am JIMS helpful bot. How can I assist you today? Are there any departments, facilities, or particular doctors you want to ask about?”
3. Affirmations: If user replies "Yes"/"ha", provide that info.
4. Symptoms & Pain: Suggest relevant departments.
5. Factual Restriction: Never use outside knowledge.
6. Organization: Use lists or tables when multiple facts exist.
7. Out-of-Scope: If user asks for information not present in JIMS data, reply:
 
   • English: "Sorry, no such information is present on the JIMS site."
   • Hindi: "माफ़ कीजिए, इस विषय में JIMS की वेबसाइट पर कोई जानकारी उपलब्ध नहीं है।"
   • Hinglish: "Maaf kijiye, JIMS site par is vishay ki jankari uplabdh nahi hai।"
8. If user asks about all departement, or all doctors reply with a list of all in the requested language.
9. If user asks about any particular name of the doctor then, reply with the details of the department that doctor is associated with.
10. If user asks about any particular department and after that asks a follow up question such as "yaha kya facilities hai?" in different language or "What are the timings of this department?" then reply with the details of that department in the same language the user has used.
=== START-JIMS-DATA ===
{context}
=== END-JIMS-DATA ===
 
Chat history:
{chat_history}
 
User: {question}
JIMS-Guide:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=base_prompt
)

# 8. Memory & chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=False,
    verbose=False
)

# 9. Response function for UI
def get_bot_response(user_input: str) -> str:
    detected_lang = detect_language(user_input)

    if detected_lang == "english":
        language_override = """
🚨 CRITICAL SYSTEM OVERRIDE 🚨
RESPONSE LANGUAGE: ENGLISH ONLY
- Use ONLY English words
- Use ONLY Latin alphabet
🚨 END OVERRIDE 🚨
"""
    elif detected_lang == "hindi":
        language_override = """
🚨 CRITICAL SYSTEM OVERRIDE 🚨  
RESPONSE LANGUAGE: HINDI ONLY
- Use ONLY Devanagari script
🚨 END OVERRIDE 🚨
"""
    else:
        language_override = """
🚨 CRITICAL SYSTEM OVERRIDE 🚨
RESPONSE LANGUAGE: HINGLISH ONLY  
- Use Latin script with Hindi vocabulary
🚨 END OVERRIDE 🚨
"""

    dynamic_prompt = language_override + base_prompt
    temp_qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=dynamic_prompt
    )
    qa_chain.combine_docs_chain.llm_chain.prompt = temp_qa_prompt

    resp = qa_chain.invoke({"question": user_input})
    bot_response = resp["answer"]

    # Language correction if mismatch
    response_lang = detect_language(bot_response)
    if response_lang != detected_lang:
        correction_prompts = {
            "english": f"Rewrite this response in pure English only:\n\n{bot_response}",
            "hindi": f"इस उत्तर को केवल हिंदी में दोबारा लिखें:\n\n{bot_response}",
            "hinglish": f"Rewrite this response in Hinglish only:\n\n{bot_response}"
        }
        corrected_response = llm.invoke(correction_prompts[detected_lang]).content
        return corrected_response

    return bot_response

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from typing import List
from dotenv import load_dotenv
import re
import time
 
load_dotenv()
 
# 1. Load & scrape data
docs = WebBaseLoader("https://ncjims.org/departments.html").load()
 
# 2. Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
 
# 3. Translation functions
def translate_documents_to_hindi(docs: List) -> List:
    translated = []
    for d in docs:
        prompt = f"Translate this medical information into Hindi (Devanagari script):\n\n{d.page_content}"
        resp = llm.invoke(prompt).content
        translated.append(type(d)(page_content=resp, metadata=d.metadata.copy()))
    return translated
 
def translate_documents_to_hinglish(docs: List) -> List:
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
embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    collection_name="jims_docs_multilang",
    persist_directory="./chroma_db"
)
 
# 6. Enhanced language detection function
def detect_language(text: str) -> str:
    # Check for Devanagari script
    if re.search(r'[\u0900-\u097F]', text):
        return "hindi"
   
    # Common Hinglish words
    hinglish_words = {"hai", "hain", "nahi", "nahin", "kya","iske", "kyun", "kaise", "kahan","kya","h"
                     "mein", "main", "aur", "ya", "par", "pe", "ko", "ka", "ke", "ki","rha","raha"
                     "se", "me", "aap", "tum", "hum", "woh", "yeh", "aise", "waise","kab", "kaun", "kitna", "kitni", "kitne", "lekin", "agar", "jab", "phir", "ab", "kyunki", "shayad", "bahut", "thoda", "zaroor", "chahiye", "kar", "bata", "puch", "mil", "namaste", "shukriya", "acha", "theek hai", "koi baat nahi", "badiya", "mast", "chalo", "suno", "samjha", "samajh gaya", "samajh gayi", "bataye", "Janna","btao", "batao", "karo", "inki", "unka", "unki", "unke", "dikhaye", "rog", "ilaj"}
   
    words = set(re.findall(r'\b\w+\b', text.lower()))
    hinglish_count = len(words.intersection(hinglish_words))
   
    if hinglish_count >= 2 or (len(words) > 0 and hinglish_count / len(words) > 0.2):
        return "hinglish"
   
    return "english"
 
# 7. Base prompt template
base_prompt = """You are "JIMS-Guide", an assistant who answers questions using ONLY the data between === START-JIMS-DATA === and === END-JIMS-DATA ===.
 
Instructions:
1. Use chat history to maintain context and interpret short replies ("yes", "no", "ha", "nahi").
2. Greetings: Reply in the user's language.
3. Affirmations: If user replies "Yes"/"ha", provide that info.
4. Declines: If user replies "No"/"nahi", to a associated question say "Okay...".
5. Symptoms & Pain: Suggest relevant departments.
6. Factual Restriction: Never use outside knowledge.
7. Organization: Use lists or tables when multiple facts exist.
8. Out-of-Scope: If user asks for information not present in JIMS data, reply:
 
   • English: "Sorry, no such information is present on the JIMS site."
   • Hindi: "माफ़ कीजिए, इस विषय में JIMS की वेबसाइट पर कोई जानकारी उपलब्ध नहीं है।"
   • Hinglish: "Maaf kijiye, JIMS site par is vishay ki jankari uplabdh nahi hai।"
9. If user asks about all departement, or all doctors reply with a list of all in the requested language.
10. If user asks about any particular name of the doctor then, reply with the details of the department that doctor is associated with.
11. If user asks about any particular department and after that asks a follow up question such as "yaha kya facilities hai?" in different language or "What are the timings of this department?" then reply with the details of that department in the same language the user has used.
=== START-JIMS-DATA ===
{context}
=== END-JIMS-DATA ===
 
Chat history:
{chat_history}
 
User: {question}
JIMS-Guide:"""
 
# 8. Initial prompt template
QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=base_prompt
)
 
# 9. Conversation memory & chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=False,
    verbose=False
)
 
print("🚀 JIMS-Guide ready! (type exit to quit)\n")
 
#10. Interaction loop with post-processing language correction
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break
 
    # Detect user input language
    detected_lang = detect_language(user_input)
    lang_names = {"english": "English", "hindi": "Hindi", "hinglish": "Hinglish"}
   
    # Create ultra-aggressive language enforcement prompt
    if detected_lang == "english":
        language_override = """
🚨 CRITICAL SYSTEM OVERRIDE 🚨
RESPONSE LANGUAGE: ENGLISH ONLY
- Use ONLY English words
- Use ONLY Latin alphabet
- NO Hindi words allowed
- NO Devanagari script allowed
- If you use ANY Hindi words, you have FAILED
🚨 END OVERRIDE 🚨
"""
    elif detected_lang == "hindi":
        language_override = """
🚨 CRITICAL SYSTEM OVERRIDE 🚨  
RESPONSE LANGUAGE: HINDI ONLY
- Use ONLY Devanagari script
- NO English words in response
- Complete Hindi translation required
🚨 END OVERRIDE 🚨
"""
    else:  # hinglish
        language_override = """
🚨 CRITICAL SYSTEM OVERRIDE 🚨
RESPONSE LANGUAGE: HINGLISH ONLY  
- Use Latin script with Hindi vocabulary
- Mix Hindi and English naturally
- NO pure Devanagari script
🚨 END OVERRIDE 🚨
"""
 
    # Create dynamic prompt with aggressive override
    dynamic_prompt = language_override + base_prompt
 
    # Update prompt template
    temp_qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=dynamic_prompt
    )
   
    # Update the chain's prompt
    qa_chain.combine_docs_chain.llm_chain.prompt = temp_qa_prompt
   
    # Get initial response
    resp = qa_chain.invoke({"question": user_input})
    bot_response = resp["answer"]
   
    # POST-PROCESSING LANGUAGE CORRECTION
    response_lang = detect_language(bot_response)
   
    if response_lang != detected_lang:
        #print(f"⚠️ Language mismatch detected! Expected: {lang_names[detected_lang]}, Got: {lang_names[response_lang]}")
       
        # Force correction with direct LLM call
        correction_prompts = {
            "english": f"Rewrite this response in pure English only, using only English words and Latin alphabet. Do not use any Hindi words:\n\n{bot_response}",
            "hindi": f"इस उत्तर को केवल हिंदी में दोबारा लिखें, केवल देवनागरी लिपि का उपयोग करें:\n\n{bot_response}",
            "hinglish": f"Rewrite this response in Hinglish only, using Latin script with Hindi vocabulary mixed naturally:\n\n{bot_response}"
        }
       
        corrected_response = llm.invoke(correction_prompts[detected_lang]).content
        print("Bot:", corrected_response)
    else:
        print("Bot:", bot_response)
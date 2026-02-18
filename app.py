import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os

# ==========================================
# 1. 頁面設定與金鑰讀取
# ==========================================
st.set_page_config(page_title="海扶及達文西問答小幫手", page_icon="🤖")
st.title("海扶及達文西問答小幫手 🤖")

# 讀取 API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 尚未設定 Google API Key。")
    st.stop()

# ==========================================
# 2. 核心修正：自動找出能用的模型名稱
# ==========================================
@st.cache_resource
def get_valid_models():
    """
    不猜測名稱，直接列出帳號可用的所有模型，並分類回傳。
    """
    chat_model = "models/gemini-pro" # 預設保底
    embed_model = "models/embedding-001" # 預設保底

    try:
        print("正在偵測可用模型...")
        # 列出所有模型
        for m in genai.list_models():
            # 找聊天模型
            if 'generateContent' in m.supported_generation_methods:
                # 優先抓 1.5 Flash 或 Pro
                if 'gemini-1.5-flash' in m.name:
                    chat_model = m.name
                elif 'gemini-1.5-pro' in m.name and 'flash' not in chat_model:
                    chat_model = m.name
            
            # 找嵌入模型 (這就是您報錯的地方)
            if 'embedContent' in m.supported_generation_methods:
                # 優先抓 text-embedding-004，抓不到就用任何一個能用的
                if 'text-embedding-004' in m.name:
                    embed_model = m.name
                elif 'embedding-001' in m.name and 'text-embedding' not in embed_model:
                    embed_model = m.name
        
        print(f"✅ 自動鎖定聊天模型: {chat_model}")
        print(f"✅ 自動鎖定嵌入模型: {embed_model}")
        return chat_model, embed_model

    except Exception as e:
        st.error(f"模型偵測失敗，將使用預設值。錯誤: {e}")
        return chat_model, embed_model

# 執行偵測
VALID_CHAT_MODEL, VALID_EMBED_MODEL = get_valid_models()

# ==========================================
# 3. 定義 Embedding (使用偵測到的模型)
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                # 使用剛剛偵測到的 VALID_EMBED_MODEL
                response = genai.embed_content(
                    model=VALID_EMBED_MODEL,
                    content=text,
                    task_type="retrieval_query"
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                # 萬一還是錯，嘗試最後一招：舊版名稱
                try:
                    response = genai.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_query"
                    )
                    embeddings.append(response['embedding'])
                except:
                    print(f"Embedding 完全失敗: {e}")
                    embeddings.append([0.0]*768) # 避免當機
        return embeddings

# ==========================================
# 4. 初始化資料庫
# ==========================================
@st.cache_resource(show_spinner="正在讀取資料...")
def initialize_vector_db():
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="medical_faq",
        embedding_function=GeminiEmbeddingFunction()
    )
    
    if collection.count() == 0:
        excel_file = "網路問答.xlsx"
        if os.path.exists(excel_file):
            try:
                data = pd.read_excel(excel_file)
                if '問題' in data.columns and '回覆' in data.columns:
                    data = data.dropna(subset=['問題', '回覆'])
                    questions = data['問題'].astype(str).tolist()
                    answers = data['回覆'].astype(str).tolist()
                    ids = [f"id-{i}" for i in range(len(questions))]
                    
                    collection.add(
                        documents=answers,
                        metadatas=[{"question": q} for q in questions],
                        ids=ids
                    )
                else:
                    st.error("Excel 格式錯誤。")
            except Exception as e:
                st.error(f"讀取 Excel 失敗: {e}")
    return collection

try:
    collection = initialize_vector_db()
except Exception as e:
    st.error(f"資料庫錯誤: {e}")
    st.stop()

# ==========================================
# 5. 聊天邏輯
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("請輸入您的醫療問題..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        results = collection.query(query_texts=[prompt], n_results=1)
        distance = results['distances'][0][0] if results['distances'] else 1.0
        best_answer = results['documents'][0][0] if results['documents'] else ""

        THRESHOLD = 0.65 

        if distance > THRESHOLD:
            final_response = (
                "這個問題比較複雜，建議您至門診進一步諮詢醫師。<br><br>"
                "<b>🏥 門診時間：</b><br>"
                "- 林口長庚醫院：週二上午、週六下午<br>"
                "- 土城醫院：週二下午、週六上午"
            )
        else:
            with st.spinner('🤖 AI 思考中...'):
                # 使用偵測到的 VALID_CHAT_MODEL
                model = genai.GenerativeModel(VALID_CHAT_MODEL)
                
                system_prompt = f"""
                你是專業的醫療助理。
                使用者問題：{prompt}
                標準答案：{best_answer}
                請根據標準答案親切回答。
                """
                
                response = model.generate_content(system_prompt)
                final_response = response.text + (
                    "<br><br>---<br>"
                    "如有疑問，歡迎 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 線上諮詢</a>。"
                )

    except Exception as e:
        final_response = f"系統發生錯誤，請稍後再試。(錯誤代碼: {e})"

    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})

# ==========================================
# 0. 系統環境修正 (SQLite Fix)
# ==========================================
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os

# ==========================================
# 1. 介面設計 (LINE 風格美化核心)
# ==========================================
st.set_page_config(page_title="海扶醫療諮詢", page_icon="🏥", layout="centered")

st.markdown("""
<style>
    /* 1. 全域字體與背景設定 (LINE 經典灰藍底色) */
    .stApp {
        background-color: #8CABD9; /* LINE 聊天室經典背景色 */
        font-family: "Microsoft JhengHei", "Heiti TC", sans-serif !important;
    }
    
    /* 2. 標題區塊 (白色底 + LINE 綠色字) */
    .header-container {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        text-align: center;
    }
    
    h1 {
        color: #00B900 !important; /* LINE 品牌綠 */
        font-weight: 900 !important;
        font-size: 28px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    p {
        font-size: 18px !important; /* 全域字體放大 */
        line-height: 1.6 !important;
    }

    /* 3. 隱藏干擾元素 */
    [data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 4. 對話氣泡優化 (卡片式設計) */
    .stChatMessage {
        background-color: #FFFFFF;
        border-radius: 20px !important; /* 更圓潤 */
        padding: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: none !important;
        margin-bottom: 15px;
    }
    
    /* 5. 連結樣式 (深綠色，加粗) */
    a {
        color: #008000 !important;
        font-weight: bold;
        text-decoration: none;
        border-bottom: 1px dashed #008000;
    }
    a:hover {
        background-color: #E8F5E9;
    }

    /* 6. 輸入框優化 */
    .stChatInputContainer {
        border-top: 1px solid #ddd;
        background-color: #ffffff;
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 標題區塊 (使用 HTML 封裝以套用樣式)
st.markdown("""
<div class="header-container">
    <h1>🏥 海扶及達文西醫療諮詢</h1>
    <div style="color: #666; font-size: 16px; margin-top: 10px;">
        陳威君醫師的 AI 小幫手<br>
        <span style="font-size: 14px; color: #999;">(24小時為您解答基本疑問)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. 系統核心邏輯 (維持不變)
# ==========================================
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 系統錯誤：未設定 API Key。")
    st.stop()

@st.cache_resource
def get_first_available_model():
    """不指定名稱，直接抓取帳號內第一個能用的模型 (盲測法)"""
    chat_model = None
    embed_model = None
    try:
        all_models = list(genai.list_models())
        # 找聊天模型
        for m in all_models:
            if 'generateContent' in m.supported_generation_methods:
                chat_model = m.name
                if 'gemini' in m.name: break 
        # 找嵌入模型
        for m in all_models:
            if 'embedContent' in m.supported_generation_methods:
                embed_model = m.name
                if 'text-embedding' in m.name: break
        return chat_model, embed_model
    except Exception as e:
        return None, None

VALID_CHAT_MODEL, VALID_EMBED_MODEL = get_first_available_model()

if not VALID_CHAT_MODEL:
    st.error("❌ 無法連線至 Google AI，請檢查 API Key 權限。")
    st.stop()

# ==========================================
# 3. 資料庫邏輯
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = genai.embed_content(
                    model=VALID_EMBED_MODEL,
                    content=text,
                    task_type="retrieval_query"
                )
                embeddings.append(response['embedding'])
            except:
                embeddings.append([0.0] * 768)
        return embeddings

@st.cache_resource(show_spinner="正在準備資料庫...")
def initialize_vector_db():
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name="medical_faq_v3",  
            embedding_function=GeminiEmbeddingFunction()
        )
        if collection.count() == 0:
            excel_file = "網路問答.xlsx"
            if os.path.exists(excel_file):
                data = pd.read_excel(excel_file).dropna(subset=['問題', '回覆'])
                questions = data['問題'].astype(str).tolist()
                answers = data['回覆'].astype(str).tolist()
                ids = [f"id-{i}" for i in range(len(questions))]
                batch_size = 20
                for i in range(0, len(questions), batch_size):
                    end = min(i + batch_size, len(questions))
                    collection.add(
                        documents=answers[i:end],
                        metadatas=[{"question": q} for q in questions[i:end]],
                        ids=ids[i:end]
                    )
        return collection
    except Exception as e:
        st.error(f"資料庫錯誤: {str(e)}")
        return None

collection = initialize_vector_db()

# ==========================================
# 4. 對話邏輯 (含頭像設定)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 歡迎訊息
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "您好！我是陳醫師的 AI 小幫手 👋<br>請問有什麼關於 **海扶刀** 或 **達文西手術** 的問題想問嗎？"
    })

# 顯示歷史訊息
for message in st.session_state.messages:
    # 設定頭像：醫師用 👨‍⚕️，使用者用 👤
    avatar = "👨‍⚕️" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)

# 輸入框
if prompt := st.chat_input("請輸入您的問題..."):
    # 顯示使用者問題
    st.chat_message("user", avatar="👤").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if collection is None:
        st.error("資料庫未啟動。")
        st.stop()
    
    final_response = ""
    
    # 搜尋與回答
    with st.spinner('🔍 正在查閱資料...'):
        try:
            results = collection.query(query_texts=[prompt], n_results=1)
            distance = results['distances'][0][0] if results['distances'] else 1.0
            best_answer = results['documents'][0][0] if results['documents'] else ""

            THRESHOLD = 0.75 

            if distance > THRESHOLD:
                final_response = (
                    "這個問題比較專業，建議您直接至門診諮詢醫師，能獲得更準確的評估喔！🏥<br><br>"
                    "<b>📅 門診時間：</b><br>"
                    "• 林口長庚：週二上午、週六下午<br>"
                    "• 土城醫院：週二下午、週六上午<br><br>"
                    "👉 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>點此聯繫官方 Line 小編</a>"
                )
            else:
                model = genai.GenerativeModel(VALID_CHAT_MODEL)
                system_prompt = f"""
                你是一位專業、親切且溫暖的婦科諮詢助理，隸屬於陳威君醫師團隊。
                【使用者問題】{prompt}
                【資料庫答案】{best_answer}
                請根據「資料庫答案」重新撰寫回覆：
                1. 語氣要像真人一樣溫暖、有同理心 (可以使用適量 emoji 如 😊, 💪)。
                2. 保持專業，內容準確。
                3. 排版要清晰，適當分段，讓手機閱讀方便。
                4. 不要提及「根據資料庫」或「標準答案」。
                """
                
                response = model.generate_content(system_prompt)
                final_response = response.text + (
                    "<br><br>---<br>"
                    "如有更多疑問，歡迎 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 線上諮詢</a> 💬"
                )

        except Exception as e:
            final_response = f"⚠️ 系統連線不穩，請稍後再試。(錯誤: {e})"

    # 顯示助手回答
    with st.chat_message("assistant", avatar="👨‍⚕️"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})

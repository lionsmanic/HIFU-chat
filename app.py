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
# 1. 介面設計 (LINE 風格 + 法規安全優化)
# ==========================================
# 網頁標籤改為 "衛教資訊"，避開 "諮詢"
st.set_page_config(page_title="海扶及達文西衛教資訊", page_icon="🏥", layout="centered")

st.markdown("""
<style>
    /* 1. 全域設定 - LINE 風格灰藍底色 */
    .stApp {
        background-color: #9bbbd4;
        font-family: "Microsoft JhengHei", "Heiti TC", sans-serif !important;
    }
    
    /* 2. 標題區塊卡片化 */
    .header-container {
        background-color: #ffffff;
        padding: 30px 20px;
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        text-align: center;
        border-top: 5px solid #2E7D32;
    }
    
    /* 3. 超大醫師頭像樣式 */
    .big-avatar {
        font-size: 70px;
        background-color: #f0f7f4;
        width: 110px;
        height: 110px;
        line-height: 110px;
        border-radius: 50%;
        margin: 0 auto 15px auto;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        border: 3px solid #ffffff;
    }
    
    /* 4. 標題字體優化 */
    .main-title {
        color: #1b5e20;
        font-weight: 900;
        font-size: 32px;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    
    .sub-title {
        color: #555;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    /* 5. 免責聲明樣式 (法規保護傘) */
    .disclaimer {
        font-size: 14px;
        color: #666;
        font-weight: 400;
        background-color: #f0f2f5;
        display: inline-block;
        padding: 8px 15px;
        border-radius: 10px;
        margin-top: 10px;
        line-height: 1.5;
        border-left: 4px solid #999;
    }

    /* 6. 隱藏 Streamlit 原生元素 */
    [data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 7. 對話氣泡優化 */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 18px !important;
        padding: 15px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: none !important;
        margin-bottom: 12px;
    }
    
    /* 8. 連結樣式 */
    a {
        color: #2E7D32 !important;
        font-weight: bold;
        text-decoration: none;
        border-bottom: 1px dashed #2E7D32;
    }
    a:hover {
        background-color: #E8F5E9;
    }
</style>
""", unsafe_allow_html=True)

# --- 標題區塊 HTML (用詞修正為衛教) ---
st.markdown("""
<div class="header-container">
    <div class="big-avatar">👨‍⚕️</div>
    <div class="main-title">海扶及達文西衛教資訊</div>
    <div class="sub-title">陳威君醫師 AI 衛教小幫手</div>
    <div class="disclaimer">
        💡 <b>本平台僅提供一般衛教知識問答</b><br>
        對話內容由 AI 輔助生成，非醫師親自即時回覆。<br>
        實際醫療狀況請務必至門診由醫師親自評估。
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
    """不指定名稱，直接抓取帳號內第一個能用的模型"""
    chat_model = None
    embed_model = None
    try:
        all_models = list(genai.list_models())
        # 1. 找聊天模型
        for m in all_models:
            if 'generateContent' in m.supported_generation_methods:
                chat_model = m.name
                if 'gemini' in m.name: break 
        # 2. 找嵌入模型
        for m in all_models:
            if 'embedContent' in m.supported_generation_methods:
                embed_model = m.name
                if 'text-embedding' in m.name: break
        return chat_model, embed_model
    except Exception:
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

@st.cache_resource(show_spinner="正在準備衛教資料庫...")
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
# 4. 對話邏輯
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 歡迎訊息 (避開 "解答" 這種絕對性字眼，改用 "提供資訊")
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "您好！我是陳醫師的 **AI 衛教小幫手** 🤖<br>我可以為您提供 **海扶刀** 或 **達文西手術** 的相關資訊與常見問答。<br><br>請輸入您想了解的主題 👇"
    })

# 顯示歷史訊息
for message in st.session_state.messages:
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
    with st.spinner('🔍 AI 正在查詢衛教資訊...'):
        try:
            results = collection.query(query_texts=[prompt], n_results=1)
            distance = results['distances'][0][0] if results['distances'] else 1.0
            best_answer = results['documents'][0][0] if results['documents'] else ""

            THRESHOLD = 0.75 

            if distance > THRESHOLD:
                final_response = (
                    "這個問題比較個別化，建議您直接至門診，由醫師親自為您評估會比較準確喔！🏥<br><br>"
                    "<b>📅 門診時間：</b><br>"
                    "• 林口長庚：週二上午、週六下午<br>"
                    "• 土城醫院：週二下午、週六上午<br><br>"
                    "👉 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>點此聯繫官方 Line 小編</a>"
                )
            else:
                model = genai.GenerativeModel(VALID_CHAT_MODEL)
                system_prompt = f"""
                你是一位專業、親切且溫暖的婦科「衛教助理」，隸屬於陳威君醫師團隊。
                
                【任務】
                根據以下資料庫內容，回答使用者的問題。
                
                【使用者問題】{prompt}
                【資料庫標準資訊】{best_answer}
                
                【回答準則】
                1. 語氣溫暖、像真人 (可使用 😊, 💪)。
                2. 僅提供「一般性衛教資訊」，避免做出具體的「醫療診斷」或「保證」。
                3. 若涉及個別病情，請溫柔提醒需回診評估。
                4. 排版清晰，適當分段。
                5. 不要提及「根據資料庫」。
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

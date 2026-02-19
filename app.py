import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os

# ==========================================
# 1. 介面設計與 CSS 美化
# ==========================================
st.set_page_config(
    page_title="海扶醫療諮詢室",
    page_icon="🏥",
    layout="centered"
)

# --- 客製化 CSS 樣式表 ---
st.markdown("""
<style>
    /* 1. 整體背景與字體 */
    .stApp {
        background-color: #fcfcfc;
        font-family: "Microsoft JhengHei", sans-serif;
    }
    
    /* 2. 標題樣式 */
    h1 {
        color: #2E7D32;
        font-weight: 700;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 15px;
    }
    
    /* 3. 隱藏側邊欄與選單 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    /* 4. 對話框優化 */
    .stChatMessage {
        border-radius: 15px;
        border: 1px solid #f0f0f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 連結顏色 */
    a { color: #2E7D32 !important; font-weight: bold; text-decoration: none; }
    a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 海扶及達文西醫療諮詢")
st.markdown(
    """<div style='text-align: center; color: #666; margin-bottom: 20px;'>
    歡迎來到陳威君醫師的 AI 諮詢室。<br>請在下方輸入您的疑問，我將為您提供初步解答。
    </div>""", 
    unsafe_allow_html=True
)

# ==========================================
# 2. 系統設定
# ==========================================

# 讀取 API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 系統錯誤：未設定 API Key。")
    st.stop()

# --- 強制設定模型 (不再自動偵測，直接指定最新版) ---
# 這是目前最穩定的組合
CHAT_MODEL = "models/gemini-1.5-flash"
EMBED_MODEL = "models/text-embedding-004"

# ==========================================
# 3. 資料庫邏輯 (含錯誤顯示)
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = genai.embed_content(
                    model=EMBED_MODEL,
                    content=text,
                    task_type="retrieval_query"
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                # 嘗試舊版模型作為備援
                try:
                    response = genai.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_query"
                    )
                    embeddings.append(response['embedding'])
                except Exception as e2:
                    print(f"Embedding Failed: {e2}")
                    embeddings.append([0.0]*768)
        return embeddings

@st.cache_resource(show_spinner="正在準備醫療資料庫...")
def initialize_vector_db():
    client = chromadb.Client()
    
    # 這裡我們使用 get_or_create 避免錯誤
    try:
        collection = client.get_or_create_collection(
            name="medical_faq",
            embedding_function=GeminiEmbeddingFunction()
        )
    except Exception as e:
        st.error(f"資料庫建立失敗: {e}")
        st.stop()
    
    # 若資料庫為空則載入
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
            except Exception as e:
                st.error(f"Excel 讀取失敗: {e}")
    return collection

try:
    collection = initialize_vector_db()
except Exception as e:
    st.error(f"系統初始化失敗: {e}")
    st.stop()

# ==========================================
# 4. 對話邏輯
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "您好，我是陳醫師的 AI 小幫手。請問有什麼我可以幫您的嗎？<br><span style='font-size:0.8em; color:#888;'>(例如：海扶刀術後多久可以上班？)</span>"
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("請輸入您的問題..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    final_response = ""
    
    # 使用 spinner 顯示溫暖的提示
    with st.spinner('🤖 醫師小幫手正在查閱資料...'):
        try:
            # 1. 搜尋
            results = collection.query(query_texts=[prompt], n_results=1)
            distance = results['distances'][0][0] if results['distances'] else 1.0
            best_answer = results['documents'][0][0] if results['documents'] else ""

            # 2. 判斷信心度 (閾值)
            THRESHOLD = 0.65 

            if distance > THRESHOLD:
                final_response = (
                    "這個問題比較複雜，建議您直接至門診諮詢醫師，以獲得最準確的評估。<br><br>"
                    "<b>🏥 門診資訊：</b><br>"
                    "• 林口長庚：週二上午、週六下午<br>"
                    "• 土城醫院：週二下午、週六上午<br><br>"
                    "💁‍♀️ 專人諮詢：<a href='https://line.me/R/ti/p/@hifudr' target='_blank'>點此聯繫 Line 小編</a>"
                )
            else:
                # 3. AI 生成
                # 這裡直接呼叫，不再 try-catch 包覆所有錯誤，以便顯示真實原因
                model = genai.GenerativeModel(CHAT_MODEL)
                
                system_prompt = f"""
                你是一位專業、親切且溫暖的婦科諮詢助理，隸屬於陳威君醫師團隊。
                【使用者問題】{prompt}
                【資料庫答案】{best_answer}
                請根據「資料庫答案」重新撰寫回覆，語氣像真人一樣溫暖，不要提及「根據資料庫」。
                """
                
                try:
                    response = model.generate_content(system_prompt)
                    final_response = response.text + (
                        "<br><br>---<br>"
                        "如有更多疑問，歡迎 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 線上諮詢</a>"
                    )
                except Exception as api_error:
                    # 如果主要模型失敗，這裡會顯示錯誤代碼
                    final_response = f"⚠️ 系統連線異常 (錯誤代碼: {api_error})。請截圖告知管理員。"

        except Exception as e:
            final_response = f"⚠️ 系統忙碌中 (錯誤代碼: {e})。請稍後再試。"

    # 顯示回覆
    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})

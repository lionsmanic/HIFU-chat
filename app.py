# ==========================================
# 0. 系統環境修正 (SQLite Fix) - 必放第一行
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
# 1. 介面設計
# ==========================================
st.set_page_config(page_title="海扶醫療諮詢室", page_icon="🏥", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #fcfcfc; font-family: "Microsoft JhengHei", sans-serif; }
    h1 { color: #2E7D32; font-weight: 700; border-bottom: 2px solid #e0e0e0; padding-bottom: 15px; }
    [data-testid="stSidebar"] {display: none;}
    .stChatMessage { border-radius: 15px; border: 1px solid #f0f0f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    a { color: #2E7D32 !important; font-weight: bold; text-decoration: none; }
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
# 2. 自動偵測可用模型 (核心修正)
# ==========================================
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 系統錯誤：未設定 API Key。")
    st.stop()

@st.cache_resource
def get_working_models():
    """
    不猜測模型名稱，直接問 Google 帳號有哪些模型可用。
    """
    chat_model = None
    embed_model = None
    
    try:
        # 列出所有模型
        all_models = list(genai.list_models())
        
        # 1. 找聊天模型
        # 優先順序: 1.5-flash -> 1.5-pro -> 1.0-pro -> 任何 gemini
        model_names = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if not model_names:
            return None, None # 帳號完全沒權限
            
        # 自動挑選最佳的
        if any('gemini-1.5-flash' in m for m in model_names):
            chat_model = next(m for m in model_names if 'gemini-1.5-flash' in m)
        elif any('gemini-1.5-pro' in m for m in model_names):
            chat_model = next(m for m in model_names if 'gemini-1.5-pro' in m)
        else:
            chat_model = model_names[0] # 有什麼用什麼

        # 2. 找嵌入模型
        embed_names = [m.name for m in all_models if 'embedContent' in m.supported_generation_methods]
        if any('text-embedding-004' in m for m in embed_names):
            embed_model = next(m for m in embed_names if 'text-embedding-004' in m)
        elif embed_names:
            embed_model = embed_names[0]
        else:
            embed_model = "models/embedding-001" # 舊版保底

        return chat_model, embed_model

    except Exception as e:
        # 如果連 list_models 都失敗，代表 Key 真的壞了
        return None, None

# 執行偵測
VALID_CHAT_MODEL, VALID_EMBED_MODEL = get_working_models()

if not VALID_CHAT_MODEL:
    st.error("❌ 嚴重錯誤：您的 API Key 無法存取任何 Google AI 模型。")
    st.warning("💡 請前往 Google AI Studio，建立一把 **「新專案 (New Project)」** 的 API Key 即可解決。")
    st.stop()

# ==========================================
# 3. 資料庫邏輯 (使用偵測到的模型)
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = genai.embed_content(
                model=VALID_EMBED_MODEL,
                content=input,
                task_type="retrieval_query"
            )
            if 'embedding' in response:
                return [response['embedding']]
            else:
                return [e for e in response['embedding']]
        except:
            return [[0.0]*768 for _ in input]

@st.cache_resource(show_spinner="正在準備醫療資料庫...")
def initialize_vector_db():
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name="medical_faq",
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
        st.error(f"資料庫初始化失敗: {str(e)}")
        return None

collection = initialize_vector_db()

# ==========================================
# 4. 對話邏輯
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "您好，我是陳醫師的 AI 小幫手。請問有什麼我可以幫您的嗎？"
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("請輸入您的問題..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if collection is None:
        st.error("資料庫未成功啟動。")
        st.stop()
    
    final_response = ""
    
    with st.spinner('🤖 醫師小幫手正在查閱資料...'):
        try:
            # 1. 搜尋
            results = collection.query(query_texts=[prompt], n_results=1)
            distance = results['distances'][0][0] if results['distances'] else 1.0
            best_answer = results['documents'][0][0] if results['documents'] else ""

            # 2. 判斷信心度
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
                # 3. AI 生成 (使用我們自動偵測到的模型 VALID_CHAT_MODEL)
                model = genai.GenerativeModel(VALID_CHAT_MODEL)
                
                system_prompt = f"""
                你是一位專業、親切且溫暖的婦科諮詢助理，隸屬於陳威君醫師團隊。
                【使用者問題】{prompt}
                【資料庫答案】{best_answer}
                請根據「資料庫答案」重新撰寫回覆：
                1. 語氣像真人一樣溫暖、有同理心。
                2. 保持專業，不要編造事實。
                3. 不要提及「根據資料庫」或「標準答案」。
                """
                
                response = model.generate_content(system_prompt)
                final_response = response.text + (
                    "<br><br>---<br>"
                    "如有更多疑問，歡迎 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 線上諮詢</a>"
                )

        except Exception as e:
            final_response = f"⚠️ 系統發生錯誤 (使用模型 {VALID_CHAT_MODEL}): {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})

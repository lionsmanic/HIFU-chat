# ==========================================
# 0. 系統環境修正 (必須放在最第一行！)
# ==========================================
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ==========================================
# 開始匯入其他套件
# ==========================================
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

st.markdown("""
<style>
    .stApp { background-color: #fcfcfc; font-family: "Microsoft JhengHei", sans-serif; }
    h1 { color: #2E7D32; font-weight: 700; border-bottom: 2px solid #e0e0e0; padding-bottom: 15px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    .stChatMessage { border-radius: 15px; border: 1px solid #f0f0f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
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
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 系統錯誤：未設定 API Key。")
    st.stop()

# ==========================================
# 3. 資料庫邏輯 (含 SQLite 修正)
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # 為了避免 Embedding 出錯導致整個當掉，這裡做雙重保險
        try:
            # 優先使用新版
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=input,
                task_type="retrieval_query"
            )
            return [response['embedding']] if 'embedding' in response else [e for e in response['embedding']] # 處理 batch 回傳格式差異
        except:
            # 備用舊版
            try:
                embeddings = []
                for text in input:
                    res = genai.embed_content(model="models/embedding-001", content=text)
                    embeddings.append(res['embedding'])
                return embeddings
            except Exception as e:
                print(f"Embedding failed: {e}")
                return [[0.0]*768 for _ in input] # 最後保底

@st.cache_resource(show_spinner="正在準備醫療資料庫...")
def initialize_vector_db():
    try:
        # 嘗試建立 Client
        client = chromadb.Client()
        
        collection = client.get_or_create_collection(
            name="medical_faq",
            embedding_function=GeminiEmbeddingFunction()
        )

        if collection.count() == 0:
            excel_file = "網路問答.xlsx"
            if os.path.exists(excel_file):
                data = pd.read_excel(excel_file)
                if '問題' in data.columns and '回覆' in data.columns:
                    data = data.dropna(subset=['問題', '回覆'])
                    questions = data['問題'].astype(str).tolist()
                    answers = data['回覆'].astype(str).tolist()
                    ids = [f"id-{i}" for i in range(len(questions))]
                    
                    # 這裡分批寫入避免一次大量請求被擋
                    batch_size = 20
                    for i in range(0, len(questions), batch_size):
                        end = min(i + batch_size, len(questions))
                        collection.add(
                            documents=answers[i:end],
                            metadatas=[{"question": q} for q in questions[i:end]],
                            ids=ids[i:end]
                        )
            else:
                st.error("找不到 Excel 檔案")
        return collection
    except Exception as e:
        # 這裡會顯示資料庫建立失敗的真實原因
        st.error(f"資料庫初始化失敗 (SQLite/ChromaDB Error): {str(e)}")
        return None

# 初始化
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
        st.error("資料庫未成功啟動，無法回答問題。")
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
                # 3. AI 生成 (嘗試多種模型名稱)
                candidates = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                success = False
                
                for model_name in candidates:
                    try:
                        model = genai.GenerativeModel(model_name)
                        system_prompt = f"""
                        你是一位專業、親切且溫暖的婦科諮詢助理。
                        【使用者問題】{prompt}
                        【資料庫答案】{best_answer}
                        請根據「資料庫答案」重新撰寫回覆，語氣像真人一樣溫暖。
                        """
                        response = model.generate_content(system_prompt)
                        final_response = response.text + (
                            "<br><br>---<br>"
                            "如有更多疑問，歡迎 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 線上諮詢</a>"
                        )
                        success = True
                        break
                    except Exception as e:
                        print(f"{model_name} failed: {e}")
                        continue
                
                if not success:
                    final_response = "⚠️ 連線失敗，請檢查 API Key 或額度。"

        except Exception as e:
            # 這次我不顯示罐頭訊息，直接顯示真實錯誤代碼給您看
            final_response = f"⚠️ 發生錯誤 (請截圖給工程師): {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})

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

# --- 客製化 CSS ---
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
# 3. 穩健型資料庫邏輯 (自動切換模型)
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # 定義備選模型清單
        model_candidates = ["models/text-embedding-004", "models/embedding-001"]
        
        embeddings = []
        for text in input:
            success = False
            for model_name in model_candidates:
                try:
                    response = genai.embed_content(
                        model=model_name,
                        content=text,
                        task_type="retrieval_query"
                    )
                    embeddings.append(response['embedding'])
                    success = True
                    break # 成功就跳出，不用試下一個
                except:
                    continue # 失敗就試下一個
            
            if not success:
                embeddings.append([0.0]*768) # 真的全掛了，回傳空向量防當機
        return embeddings

@st.cache_resource(show_spinner="正在準備醫療資料庫...")
def initialize_vector_db():
    client = chromadb.Client()
    try:
        collection = client.get_or_create_collection(
            name="medical_faq",
            embedding_function=GeminiEmbeddingFunction()
        )
    except:
        # 如果無法建立，嘗試重置
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        client = chromadb.Client()
        collection = client.create_collection(
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
                    collection.add(documents=answers, metadatas=[{"question": q} for q in questions], ids=ids)
            except:
                pass
    return collection

try:
    collection = initialize_vector_db()
except:
    st.error("系統初始化異常，請重新整理頁面。")
    st.stop()

# ==========================================
# 4. 對話邏輯 (核心修正：生成模型的自動降級)
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
                # 3. AI 生成 (自動降級邏輯)
                # 這裡定義一串模型，優先試 1.5-flash，不行試 1.5-pro，再不行試 gemini-pro (舊版)
                chat_candidates = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
                
                generated_text = ""
                
                for model_name in chat_candidates:
                    try:
                        model = genai.GenerativeModel(model_name)
                        system_prompt = f"""
                        你是一位專業、親切且溫暖的婦科諮詢助理，隸屬於陳威君醫師團隊。
                        【使用者問題】{prompt}
                        【資料庫答案】{best_answer}
                        請根據「資料庫答案」重新撰寫回覆，語氣像真人一樣溫暖，不要提及「根據資料庫」。
                        """
                        response = model.generate_content(system_prompt)
                        generated_text = response.text
                        # 成功產生文字，就跳出迴圈
                        break 
                    except Exception as e:
                        # 記錄錯誤但繼續嘗試下一個模型
                        print(f"Model {model_name} failed: {e}")
                        continue
                
                if generated_text:
                    final_response = generated_text + (
                        "<br><br>---<br>"
                        "如有更多疑問，歡迎 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 線上諮詢</a>"
                    )
                else:
                    # 如果所有模型都失敗
                    final_response = "⚠️ 目前 AI 系統連線忙碌，請稍後再試，或直接聯繫 Line 小編。"

        except Exception as e:
            final_response = f"⚠️ 系統發生未知錯誤。請稍後再試。"

    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})

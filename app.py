import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os

# ==========================================
# 1. 介面設計與 CSS 美化 (UI/UX 重點)
# ==========================================
st.set_page_config(
    page_title="海扶醫療諮詢室",
    page_icon="🏥",
    layout="centered"
)

# --- 客製化 CSS 樣式表 ---
st.markdown("""
<style>
    /* 1. 整體背景與字體設定 */
    .stApp {
        background-color: #f9f9f9; /* 柔和灰白底色 */
        font-family: "Microsoft JhengHei", "Helvetica", sans-serif;
    }
    
    /* 2. 標題樣式 */
    h1 {
        color: #2E7D32; /* 醫學綠 */
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* 3. 隱藏預設選單與側邊欄 (讓介面更乾淨) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    /* 4. 對話框優化 */
    /* 使用者對話框 */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #4CAF50;
    }
    
    /* 助理對話框背景優化 */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        border: 1px solid #f0f0f0;
    }

    /* 連結顏色 */
    a {
        color: #2E7D32 !important;
        font-weight: bold;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    
    /* 輸入框區域優化 */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 標題區
st.title("🏥 海扶及達文西醫療諮詢")
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-bottom: 30px; font-size: 1.1em;'>
    歡迎來到陳威君醫師的 AI 諮詢室。<br>
    請在下方輸入您的疑問，我將為您提供初步解答。
    </div>
    """, 
    unsafe_allow_html=True
)

# ==========================================
# 2. 系統設定 (後台運作，不顯示給使用者)
# ==========================================

# 讀取 API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("系統維護中 (API Key Missing)，請稍後再試。")
    st.stop()

# --- 靜默模型選擇器 (不再顯示文字) ---
@st.cache_resource
def get_best_models_silently():
    """後台自動挑選最佳模型，不報錯，不顯示"""
    chat_model = "models/gemini-pro"
    embed_model = "models/embedding-001"
    
    try:
        # 取得所有可用模型
        all_models = [m for m in genai.list_models()]
        
        # 1. 挑選聊天模型 (優先順序: 1.5-Flash -> 1.5-Pro -> 1.0-Pro)
        chat_candidates = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if any('gemini-1.5-flash' in m for m in chat_candidates):
            chat_model = next(m for m in chat_candidates if 'gemini-1.5-flash' in m)
        elif any('gemini-1.5-pro' in m for m in chat_candidates):
            chat_model = next(m for m in chat_candidates if 'gemini-1.5-pro' in m)
        elif chat_candidates:
            chat_model = chat_candidates[0] # 隨便選一個能用的
            
        # 2. 挑選嵌入模型 (優先順序: text-embedding-004 -> embedding-001)
        embed_candidates = [m.name for m in all_models if 'embedContent' in m.supported_generation_methods]
        
        if any('text-embedding-004' in m for m in embed_candidates):
            embed_model = next(m for m in embed_candidates if 'text-embedding-004' in m)
        elif embed_candidates:
            embed_model = embed_candidates[0]
            
    except:
        pass # 發生任何錯誤都使用預設值
    
    return chat_model, embed_model

# 執行靜默偵測
CHAT_MODEL, EMBED_MODEL = get_best_models_silently()

# ==========================================
# 3. 資料庫邏輯
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
            except:
                embeddings.append([0.0]*768)
        return embeddings

@st.cache_resource(show_spinner="正在準備醫療資料庫...")
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
            except:
                pass 
    return collection

try:
    collection = initialize_vector_db()
except:
    st.error("資料庫連線異常，請重新整理頁面。")
    st.stop()

# ==========================================
# 4. 對話邏輯
# ==========================================

# 初始化訊息
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 可以加入一個歡迎訊息
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "您好，我是陳醫師的 AI 小幫手。請問有什麼我可以幫您的嗎？<br>(例如：海扶刀術後多久可以上班？)"
    })

# 顯示歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# 接收使用者輸入 (位於底部是 Streamlit 的標準設計，適合手機操作)
if prompt := st.chat_input("請輸入您的問題..."):
    # 顯示使用者問題
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    final_response = ""
    
    # 搜尋與生成 (使用更人性化的提示文字)
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
                # 3. AI 生成 (使用先前靜默偵測到的模型)
                model = genai.GenerativeModel(CHAT_MODEL)
                
                system_prompt = f"""
                你是一位專業、親切且溫暖的婦科諮詢助理，隸屬於陳威君醫師團隊。
                
                【使用者問題】{prompt}
                【資料庫答案】{best_answer}
                
                請根據「資料庫答案」重新撰寫回覆：
                1. 語氣要像真人一樣溫暖、有同理心，不要像機器人。
                2. 保持專業，不要編造事實。
                3. 排版要清晰易讀 (適當分段)。
                4. 不要提到「根據資料庫」或「標準答案」這類字眼，直接回答即可。
                """
                
                response = model.generate_content(system_prompt)
                final_response = response.text + (
                    "<br><br>---<br>"
                    "如有更多疑問，歡迎 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 線上諮詢</a>"
                )
                
        except Exception:
            final_response = "抱歉，系統網路忙碌中，請稍後再試，或直接聯繫 Line 小編。"

    # 顯示回覆
    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})

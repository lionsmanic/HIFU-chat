import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os

# ==========================================
# 1. 頁面設定與金鑰讀取
# ==========================================
st.set_page_config(
    page_title="海扶及達文西問答小幫手",
    page_icon="🤖",
    layout="centered"
)

st.title("海扶及達文西問答小幫手 🤖")
st.markdown("輸入問題，即可獲得專業回覆！如果仍有疑問，可透過 Line 進一步諮詢。")

# 嘗試從 Streamlit Secrets 讀取 API Key
# 本地開發請在 .streamlit/secrets.toml 設定
# 雲端部署請在 Streamlit Cloud 的 Secrets 設定
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 尚未設定 Google API Key。請在 Secrets 中設定 'GOOGLE_API_KEY'。")
    st.stop()

# ==========================================
# 2. 定義 Gemini Embedding 函數 (給 ChromaDB 用)
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/text-embedding-004"
        embeddings = []
        # 為了避免速率限制，這裡逐筆處理，若資料量大可考慮 batch 處理
        for text in input:
            try:
                response = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_query"
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                # 簡單錯誤處理，避免單一失敗卡死
                print(f"Embedding error: {e}")
                embeddings.append([0]*768) # 回傳空向量作為 fallback
        return embeddings

# ==========================================
# 3. 初始化資料庫 (使用快取，只執行一次)
# ==========================================
@st.cache_resource
def initialize_vector_db():
    # 使用 ephemeral client (記憶體模式)，適合 Streamlit Cloud 環境
    # 如果資料量大，建議每次重啟時重新建立索引
    client = chromadb.Client()
    
    try:
        # 嘗試取得集合
        collection = client.get_collection(
            name="medical_faq",
            embedding_function=GeminiEmbeddingFunction()
        )
    except ValueError:
        # 若不存在則建立
        collection = client.create_collection(
            name="medical_faq",
            embedding_function=GeminiEmbeddingFunction()
        )
        
        # 讀取 Excel 資料
        excel_file = "網路問答.xlsx"
        if os.path.exists(excel_file):
            try:
                data = pd.read_excel(excel_file)
                # 確保欄位存在
                if '問題' in data.columns and '回覆' in data.columns:
                    # 移除空值
                    data = data.dropna(subset=['問題', '回覆'])
                    
                    questions = data['問題'].astype(str).tolist()
                    answers = data['回覆'].astype(str).tolist()
                    ids = [f"id-{i}" for i in range(len(questions))]
                    
                    # 寫入 ChromaDB
                    collection.add(
                        documents=answers,     # 搜尋內容 (回覆)
                        metadatas=[{"question": q} for q in questions], # 關聯問題
                        ids=ids
                    )
                    print(f"成功載入 {len(questions)} 筆問答資料。")
                else:
                    st.error("Excel 檔案格式錯誤：找不到 '問題' 或 '回覆' 欄位。")
            except Exception as e:
                st.error(f"讀取 Excel 失敗: {e}")
        else:
            st.warning(f"找不到 '{excel_file}'，請確認檔案已上傳至 GitHub。")
            
    return collection

# 執行初始化
collection = initialize_vector_db()

# ==========================================
# 4. 聊天視窗邏輯
# ==========================================

# 初始化聊天歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# 處理使用者輸入
if prompt := st.chat_input("請輸入您的醫療問題..."):
    # 1. 顯示使用者訊息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 搜尋最相似的問答
    results = collection.query(
        query_texts=[prompt],
        n_results=1
    )

    # 取得距離與答案
    # Chroma 預設使用 L2 距離 (越小越相似)
    distance = results['distances'][0][0] if results['distances'] else 1.0
    best_answer = results['documents'][0][0] if results['documents'] else ""

    # 3. 判斷邏輯
    final_response = ""
    
    # === 信心門檻設定 (可微調) ===
    # 距離 > THRESHOLD 代表找不到夠像的答案
    THRESHOLD = 0.65 

    if distance > THRESHOLD:
        # 信心不足，回傳門診資訊
        final_response = (
            "這個問題比較複雜，建議您至門診進一步諮詢醫師，以獲得最準確的評估。<br><br>"
            "<b>🏥 門診時間：</b><br>"
            "- 林口長庚醫院：週二上午、週六下午<br>"
            "- 土城醫院：週二下午、週六上午<br><br>"
            "如果您有更多疑問，也歡迎透過 "
            "<a href='https://line.me/R/ti/p/@hifudr' target='_blank' style='color: #4CAF50; font-weight: bold; text-decoration: none;'>Line 小編</a> "
            "進一步線上諮詢哦！"
        )
    else:
        # 信心足夠，呼叫 Gemini 生成溫暖回覆
        try:
            chat_model = genai.GenerativeModel('gemini-1.5-flash')
            
            system_prompt = f"""
            你是一位專業且溫暖的婦科醫療諮詢助理，隸屬於陳威君醫師團隊。
            
            【任務目標】
            使用者的問題是：{prompt}
            資料庫檢索到的標準答案是：{best_answer}
            
            請根據「標準答案」的內容，用更口語、親切、像真人的方式回答使用者。
            
            【回答準則】
            1. 語氣要溫柔、有同理心，讓患者感到安心。
            2. 內容必須準確基於標準答案，不可自行編造醫學事實。
            3. 結尾可以適當加上鼓勵的話。
            4. 保持簡潔，不要過度冗長。
            """
            
            response = chat_model.generate_content(system_prompt)
            gpt_reply = response.text
            
            # 加上 Line 連結 footer
            final_response = gpt_reply + (
                "<br><br>---<br>"
                "如果您有更多疑問，歡迎透過 "
                "<a href='https://line.me/R/ti/p/@hifudr' target='_blank' style='color: #4CAF50; font-weight: bold; text-decoration: none;'>Line 小編</a> "
                "線上諮詢。"
            )
            
        except Exception as e:
            final_response = f"抱歉，系統暫時繁忙，請稍後再試。(錯誤代碼: {e})"

    # 4. 顯示並儲存助手回覆
    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})

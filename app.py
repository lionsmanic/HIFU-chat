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

# 讀取 API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 尚未設定 Google API Key。請在 Secrets 中設定 'GOOGLE_API_KEY'。")
    st.stop()

# ==========================================
# 2. 定義 Gemini Embedding 函數
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/text-embedding-004"
        embeddings = []
        for text in input:
            try:
                response = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_query"
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                # 若發生錯誤，回傳全零向量避免當機
                print(f"Embedding error: {e}")
                embeddings.append([0.0]*768)
        return embeddings

# ==========================================
# 3. 初始化資料庫 (修正版)
# ==========================================
@st.cache_resource
def initialize_vector_db():
    # 使用 ephemeral client (記憶體模式)
    client = chromadb.Client()
    
    # --- 修正重點：改用 get_or_create_collection ---
    # 這個方法會自動判斷：如果資料庫不存在就建立，存在就讀取
    # 這樣就不會因為找不到資料庫而報錯了
    collection = client.get_or_create_collection(
        name="medical_faq",
        embedding_function=GeminiEmbeddingFunction()
    )
    
    # 判斷資料庫是否為空 (count == 0 代表剛建立或是空的)
    if collection.count() == 0:
        excel_file = "網路問答.xlsx"
        if os.path.exists(excel_file):
            try:
                data = pd.read_excel(excel_file)
                if '問題' in data.columns and '回覆' in data.columns:
                    # 移除空值
                    data = data.dropna(subset=['問題', '回覆'])
                    
                    questions = data['問題'].astype(str).tolist()
                    answers = data['回覆'].astype(str).tolist()
                    ids = [f"id-{i}" for i in range(len(questions))]
                    
                    # 寫入 ChromaDB
                    collection.add(
                        documents=answers,
                        metadatas=[{"question": q} for q in questions],
                        ids=ids
                    )
                    print(f"✅ 成功載入 {len(questions)} 筆問答資料。")
                else:
                    st.error("❌ Excel 檔案格式錯誤：找不到 '問題' 或 '回覆' 欄位。")
            except Exception as e:
                st.error(f"❌ 讀取 Excel 失敗: {e}")
        else:
            st.warning(f"⚠️ 找不到 '{excel_file}'，請確認檔案已上傳至 GitHub。")
            
    return collection

# 執行初始化
collection = initialize_vector_db()

# ==========================================
# 4. 聊天視窗邏輯
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
        results = collection.query(
            query_texts=[prompt],
            n_results=1
        )

        distance = results['distances'][0][0] if results['distances'] else 1.0
        best_answer = results['documents'][0][0] if results['documents'] else ""

        # === 信心門檻 (可調整) ===
        THRESHOLD = 0.65 

        if distance > THRESHOLD:
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
            4. 保持簡潔。
            """
            
            response = chat_model.generate_content(system_prompt)
            final_response = response.text + (
                "<br><br>---<br>"
                "如果您有更多疑問，歡迎透過 "
                "<a href='https://line.me/R/ti/p/@hifudr' target='_blank' style='color: #4CAF50; font-weight: bold; text-decoration: none;'>Line 小編</a> "
                "線上諮詢。"
            )

    except Exception as e:
        final_response = f"抱歉，系統暫時繁忙，請稍後再試。(錯誤: {e})"

    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})

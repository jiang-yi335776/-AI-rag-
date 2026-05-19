import os.path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# 加载环境变量
load_dotenv()

# 阿里百炼 Embedding
EMBADDINGS = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

DB_DIR = 'faiss_db/'


def save_vectors_db():

    if os.path.exists(DB_DIR):
        print('向量数据库已经构建')
    else:
        with open('sales_datas.txt', encoding='utf8') as f:
            contents = f.read()

        # 文本切分
        text_splitter = CharacterTextSplitter(
            separator=r'\d+\.\n',
            is_separator_regex=True,
            chunk_size=100,
            chunk_overlap=0,
            length_function=len
        )

        docs = text_splitter.create_documents([contents])

        print("文档数量:", len(docs))

        # 创建向量库
        db = FAISS.from_documents(docs, EMBADDINGS)

        db.save_local(DB_DIR)

        # result = db.similarity_search(
        #     "你好，我想知道这个小区吵不吵？"
        # )

        # print(result)

def init_chain():
    """得到一个chain"""
    #第一步：加载向量数据库
    db = FAISS.load_local(DB_DIR, EMBADDINGS,allow_dangerous_deserialization=True)

    #第二步：创建提示词模板

    #创建一个问题模板
    system_prompt =  """
    你是一个房地产销售问答助手。

    你只能根据下面的上下文回答用户问题。
    如果上下文中没有明确提到答案，必须只回答：“这个问题，我建议你直接问人工！”。
    不要使用你自己的常识、经验或推测。
    不要编造答案。

    上下文：
    {context}
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )



    #第三步：创建一个链

    #创建一个搜索器:similarity_score_threshold:根据相似度的分数来返回结果。“score_threshold”:0.8:分值>=0.7
    retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.2
    }
    )
    model = ChatTongyi(
    model="qwen-max",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY",),
    temperature=0.2,
)
    #将检索到的结果（多个docs）输入到提示模板中
    chain1 = create_stuff_documents_chain(llm=model,prompt=prompt_template)
    chain = create_retrieval_chain(retriever=retriever,combine_docs_chain=chain1)
    return chain




if __name__ == '__main__':
    save_vectors_db()
    chain = init_chain()
    res =  chain.invoke({'input': '小区吵不吵'})
    print(res)

# -*- coding: utf-8 -*-
# pip install streamlit llama-index chromadb

import json
import time
import os
from urllib import response
from dotenv import load_dotenv

from pathlib import Path
from typing import List, Dict

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
import streamlit as st


QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个专业的法律助手，请严格根据以下法律条文回答问题：\n"
    "相关法律条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response_template = PromptTemplate(QA_TEMPLATE)

# ===============  加载环境变量 ===============
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# ================== 配置区 ==================
class Config:
    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./chroma_db"
    PERSIST_DIR = "./storage"

    COLLECTION_NAME = "chinese_labor_laws"
    TOP_K = 3

    # OpenAI 模型配置
    LLM_MODEL = "gpt-4o-mini"
    
    # Embedding 模型配置
    EMBED_MODEL = "text-embedding-3-small"

# ================== 初始化模型 ==================
def init_models():
    """初始化 OpenAI 模型并验证"""

    # Embedding 模型（OpenAI）
    embed_model = OpenAIEmbedding(
        model=Config.EMBED_MODEL,
        api_key = os.getenv("OPENAI_API_KEY"), 
        api_base = os.getenv("OPENAI_BASE_URL")
        )

    # LLM 模型（OpenAI）
    llm = OpenAI(
        model=Config.LLM_MODEL,
        api_key = os.getenv("OPENAI_API_KEY"), 
        api_base = os.getenv("OPENAI_BASE_URL"),
        temperature=0
        )

    Settings.embed_model = embed_model
    Settings.llm = llm

    # 验证 embedding
    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")

    return embed_model, llm

# ================== 数据处理 ==================
# 层级字段：编/分编/章/节（无对应层级时为空字符串）
HIERARCHY_FIELDS = ("book", "sub_book", "chapter", "section")

def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件

    数据格式：列表，每个元素是一条法条，必须包含 title 和 content 字符串字段，
    可选层级字段 book/sub_book/chapter/section（见 HIERARCHY_FIELDS）。
    """
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"未找到JSON文件于 {data_dir}"

    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                for idx, item in enumerate(data):
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 第 {idx} 条不是字典")
                    for field in ("title", "content"):
                        if not isinstance(item.get(field), str) or not item[field]:
                            raise ValueError(f"文件 {json_file.name} 第 {idx} 条缺少非空的字符串字段 '{field}'")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 第 {idx} 条中字段 '{k}' 的值不是字符串")
                    all_data.append({
                        "article": item,
                        "source": json_file.name
                    })
            except Exception as e:
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")

    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data

def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    nodes = []
    for entry in raw_data:
        article = entry["article"]
        source_file = entry["source"]

        full_title = article["title"]
        node_id = f"{source_file}::{full_title}"
        parts = full_title.split(" ", 1)
        law_name = parts[0] if len(parts) > 0 else "未知法律"
        article_no = parts[1] if len(parts) > 1 else "未知条款"

        metadata = {
            "law_name": law_name,
            "article": article_no,
            "full_title": full_title,
            "source_file": source_file,
            "content_type": "legal_article"
        }
        # 层级元数据（编/分编/章/节），缺省为空字符串
        for field in HIERARCHY_FIELDS:
            metadata[field] = article.get(field, "")

        node = TextNode(
            text=article["content"],
            id_=node_id,
            metadata=metadata
        )
        nodes.append(node)

    print(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
    return nodes

# ================== 向量存储 ==================
def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    if chroma_collection.count() == 0 and nodes is not None:
        print(f"创建新索引（{len(nodes)}个节点）...")
        storage_context.docstore.add_documents(nodes)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        print("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    print("\n存储验证结果：")
    doc_count = len(storage_context.docstore.docs)
    print(f"DocStore记录数：{doc_count}")
    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"示例节点ID：{sample_key}")
    else:
        print("警告：文档存储为空，请检查节点添加逻辑！")

    return index

# ================== Streamlit 前端 ==================
def main():
    st.title("法律智能助手")

    embed_model, llm = init_models()

    if not Path(Config.VECTOR_DB_DIR).exists():
        st.write("初始化数据...")
        raw_data = load_and_validate_json_files(Config.DATA_DIR)
        nodes = create_nodes(raw_data)
    else:
        nodes = None

    st.write("初始化向量存储...")
    start_time = time.time()
    index = init_vector_store(nodes)
    st.write(f"索引加载耗时：{time.time()-start_time:.2f}s")

    query_engine = index.as_query_engine(
        similarity_top_k=Config.TOP_K,
        text_qa_template=response_template,
        verbose=True
    )

        # 初始化 session_state
    if "input_question" not in st.session_state:
        st.session_state.input_question = ""
    
    # 用户输入
    question = st.text_input("请输入民法相关问题（输入q退出）: ", value=st.session_state.input_question, key="input_question") #需要将用户输入和input_question这个key关联，以便后面通过st.session_state.clear()整个清除
    #question = st.text_input("请输入劳动法相关问题（输入q退出）: ") 

    # @st.fragment
    # def clear_results():
    #     if st.button("清除查询结果"):
    #         st.session_state.clear()  # 清除所有会话状态。如果有清除用户问题，就不要清除整个session_state
    #         st.rerun()  # 重新运行应用以清除界面

    # clear_results()

    @st.fragment
    def clear_results():
        # st.session_state.input_question = ""  # 强制清空输入框程序会报错，不允许单独清除已经实例化的input_question
        st.session_state.clear()  # 清除所有会话状态。如果有清除用户问题，就不要清除整个session_state
    
    st.button("清除查询结果", on_click=clear_results)
    
    if question.strip() and question.lower() != 'q':
        # 执行查询
        response = query_engine.query(question)
        
        # 显示结果
        st.write(f"\n智能助手回答：\n{response.response}")
        st.write("\n支持依据：")
        for idx, node in enumerate(response.source_nodes, 1):
            meta = node.metadata
            st.write(f"\n[{idx}] {meta['full_title']}")
            hierarchy = " / ".join(filter(None, (meta.get(f, "") for f in HIERARCHY_FIELDS)))
            if hierarchy:
                st.write(f"  章节定位：{hierarchy}")
            st.write(f"  来源文件：{meta['source_file']}")
            st.write(f"  法律名称：{meta['law_name']}")
            st.write(f"  条款内容：{node.text[:100]}...")

            if hasattr(node, 'score') and node.score is not None:  # 确保 node 对象有 score 属性 并 检查 score 是否为 None
                st.write(f"  相关度得分：{node.score:.4f}")
            else:
                st.write("  相关度得分：无")

if __name__ == "__main__":
    main()

"""
RAG (检索增强生成) 完整演示

这个文件整合了三种向量数据库，展示完整的RAG流程

RAG 工作原理：
1. 准备知识库：把文档切分成小块，转换成向量存储
2. 检索：用户提问时，找到最相关的文档块
3. 生成：把相关文档和问题一起发给大模型生成答案

向量数据库对比：
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   特性       │ ChromaDB    │ FAISS       │ Milvus Lite │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 易用性       │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐     │ ⭐⭐⭐⭐   │
│ 性能         │ ⭐⭐⭐     │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐   │
│ 功能丰富度   │ ⭐⭐⭐     │ ⭐⭐       │ ⭐⭐⭐⭐⭐ │
│ 适用场景     │ 原型开发     │ 大规模检索   │ 生产环境    │
│ 是否需要服务器│ 否         │ 否          │ Lite版不需要 │
└─────────────┴─────────────┴─────────────┴─────────────┘
"""

import os  # 操作系统相关功能
from sentence_transformers import SentenceTransformer  # embedding模型

# 导入三种向量数据库
from simple_vector_db import SimpleVectorDB
from chromadb_demo import ChromaDBDemo
from faiss_demo import FAISSDemo


def print_separator(title: str):
    """打印分隔线，美化输出"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_sample_documents() -> list:
    """
    加载示例文档
    
    这些文档模拟一个简单的知识库
    实际应用中，这些来自PDF、网页等
    """
    documents = [
        # Python相关
        "Python是一种高级编程语言，由Guido van Rossum于1991年创建。Python的设计哲学强调代码的可读性和简洁性。",
        "Python支持多种编程范式，包括面向对象、函数式和过程式编程。它拥有丰富的标准库和第三方库。",
        "Python常用的库包括NumPy（数值计算）、Pandas（数据分析）、Matplotlib（数据可视化）等。",
        
        # 机器学习相关
        "机器学习是人工智能的一个分支，它让计算机能够从数据中学习，而不需要明确编程。",
        "机器学习主要分为三类：监督学习（有标签数据）、无监督学习（无标签数据）和强化学习（通过奖励学习）。",
        "常用的机器学习算法包括：线性回归、决策树、随机森林、支持向量机（SVM）、K近邻（KNN）等。",
        
        # 深度学习相关
        "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂模式。",
        "常见的深度学习框架有TensorFlow（Google开发）、PyTorch（Facebook开发）和Keras（高级API）。",
        "深度学习的典型应用包括：图像识别、语音识别、自然语言处理和游戏AI等。",
        
        # 向量数据库相关
        "向量数据库是专门用于存储和检索高维向量的数据库，常用于AI应用中的相似性搜索。",
        "向量数据库的核心技术是近似最近邻搜索（ANN），它能够在海量向量中快速找到最相似的向量。",
        "常见的向量数据库包括：ChromaDB、FAISS、Milvus、Pinecone、Weaviate等。",
        
        # RAG相关
        "RAG（检索增强生成）是一种结合检索和生成的技术，能够提高大语言模型的回答准确性。",
        "RAG的工作流程：首先将文档切分成小块并转换为向量，然后检索相关文档块，最后将这些文档块和问题一起输入大模型。",
        "RAG的优势在于：减少幻觉（hallucination）、知识可更新、可追溯来源。",
    ]
    
    return documents


def demo_with_simple_db(documents: list):
    """
    使用简易向量数据库的RAG演示
    
    这个演示帮助理解RAG的基本原理
    """
    print_separator("简易向量数据库 RAG 演示")
    
    # 创建简易数据库实例
    db = SimpleVectorDB()
    
    # 加载embedding模型
    print("\n[1/4] 加载embedding模型...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 将文档转换为向量并存储
    print("[2/4] 将文档转换为向量并存储...")
    embeddings = model.encode(documents)
    
    for i, (text, embedding) in enumerate(zip(documents, embeddings)):
        db.add(text, embedding)
    
    print(f"已存储 {db.count()} 条文档")
    
    # 用户提问
    query = "什么是深度学习？"
    print(f"\n[3/4] 用户提问: {query}")
    
    # 将问题转换为向量
    query_embedding = model.encode(query)
    
    # 搜索相关文档
    print("[4/4] 搜索相关文档...")
    results = db.search(query_embedding, top_k=3)
    
    # 显示结果
    print("\n" + "-" * 40)
    print("检索到的相关文档：")
    print("-" * 40)
    
    for i, (id, text, score) in enumerate(results):
        print(f"\n[{i+1}] 相似度: {score:.4f}")
        print(f"    {text[:80]}...")
    
    # 模拟生成答案（实际应用中调用大模型）
    print("\n" + "-" * 40)
    print("模拟生成答案：")
    print("-" * 40)
    context = "\n".join([text for _, text, _ in results])
    print(f"[RAG上下文]\n{context}\n")
    print("[答案] 深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂模式。")
    print("常见的深度学习框架有TensorFlow和PyTorch，应用包括图像识别、语音识别等。")


def demo_with_chromadb(documents: list):
    """
    使用ChromaDB的RAG演示
    
    ChromaDB会自动处理embedding转换
    """
    print_separator("ChromaDB RAG 演示")
    
    # 创建ChromaDB实例
    db = ChromaDBDemo("rag_knowledge_base")
    
    # 添加文档（ChromaDB自动处理embedding）
    print("\n[1/3] 添加文档到ChromaDB...")
    db.add_documents(documents)
    
    # 搜索相关文档
    query = "机器学习有哪些算法？"
    print(f"\n[2/3] 用户提问: {query}")
    
    print("[3/3] 搜索相关文档...")
    results = db.search(query, top_k=3)
    
    # 显示结果
    print("\n" + "-" * 40)
    print("检索到的相关文档：")
    print("-" * 40)
    
    for i, result in enumerate(results):
        print(f"\n[{i+1}] 距离: {result['distance']:.4f}")
        print(f"    {result['text'][:80]}...")


def demo_with_faiss(documents: list):
    """
    使用FAISS的RAG演示
    
    FAISS性能最高，适合大规模数据
    """
    print_separator("FAISS RAG 演示")
    
    # 创建FAISS实例
    db = FAISSDemo(dimension=384, index_type="flat")
    
    # 添加文档
    print("\n[1/3] 添加文档到FAISS...")
    db.add_documents(documents)
    
    # 搜索相关文档
    query = "什么是向量数据库？"
    print(f"\n[2/3] 用户提问: {query}")
    
    print("[3/3] 搜索相关文档...")
    results = db.search(query, top_k=3)
    
    # 显示结果
    print("\n" + "-" * 40)
    print("检索到的相关文档：")
    print("-" * 40)
    
    for i, result in enumerate(results):
        print(f"\n[{i+1}] 距离: {result['distance']:.4f}")
        print(f"    {result['text'][:80]}...")


def compare_search_speed():
    """
    对比三种数据库的搜索速度
    
    注意：这只是简单对比，实际性能取决于很多因素
    """
    print_separator("搜索速度对比")
    
    import time  # 时间库，用于计时
    import numpy as np
    
    # 准备测试数据
    num_vectors = 1000  # 测试1000个向量
    dimension = 384
    
    print(f"\n测试规模: {num_vectors} 个向量，维度 {dimension}")
    
    # 生成随机向量
    vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
    query_vector = np.random.rand(1, dimension).astype(np.float32)
    
    # 测试简易数据库
    simple_db = SimpleVectorDB()
    for i, vec in enumerate(vectors):
        simple_db.add(f"doc_{i}", vec)
    
    start = time.time()
    for _ in range(100):  # 搜索100次
        simple_db.search(query_vector[0], top_k=5)
    simple_time = (time.time() - start) / 100
    
    # 测试FAISS
    import faiss
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(vectors)
    
    start = time.time()
    for _ in range(100):
        faiss_index.search(query_vector, 5)
    faiss_time = (time.time() - start) / 100
    
    # 显示结果
    print(f"\n搜索100次平均耗时：")
    print(f"  简易数据库: {simple_time*1000:.2f} ms")
    print(f"  FAISS:      {faiss_time*1000:.2f} ms")
    print(f"\nFAISS比简易数据库快: {simple_time/faiss_time:.1f} 倍")


def main():
    """
    主函数：运行所有演示
    """
    print("\n" + "🎓" * 30)
    print("\n    RAG (检索增强生成) 学习演示")
    print("\n" + "🎓" * 30)
    
    # 加载示例文档
    print("\n正在加载示例文档...")
    documents = load_sample_documents()
    print(f"共加载 {len(documents)} 条文档")
    
    # 演示1：简易向量数据库
    demo_with_simple_db(documents)
    
    # 演示2：ChromaDB
    demo_with_chromadb(documents)
    
    # 演示3：FAISS
    demo_with_faiss(documents)
    
    # 演示4：速度对比
    compare_search_speed()
    
    # 总结
    print_separator("学习总结")
    print("""
📚 向量数据库核心概念：

1. 向量化（Embedding）
   - 将文本转换为数字向量（一串数字）
   - 相似文本的向量在空间中距离近
   
2. 索引（Index）
   - 加速搜索的数据结构
   - 常见类型：暴力搜索、倒排索引、HNSW图索引
   
3. 相似度搜索
   - 找到与查询向量最相似的向量
   - 常用度量：余弦相似度、欧几里得距离

📊 三种数据库对比：

┌─────────────┬──────────────────────────────────────────────┐
│ ChromaDB    │ 最简单，适合学习和原型开发                     │
├─────────────┼──────────────────────────────────────────────┤
│ FAISS       │ 性能最高，适合大规模数据                       │
├─────────────┼──────────────────────────────────────────────┤
│ Milvus Lite │ 功能最全，支持过滤查询，适合生产环境           │
└─────────────┴──────────────────────────────────────────────┘

💡 下一步学习建议：

1. 尝试修改代码，观察不同的搜索结果
2. 添加更多文档，测试搜索效果
3. 尝试不同的embedding模型
4. 学习如何连接真正的大模型生成答案
""")


# 程序入口
if __name__ == "__main__":
    main()

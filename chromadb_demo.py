"""
ChromaDB 向量数据库示例

ChromaDB 特点：
1. 最简单易用的向量数据库
2. 纯Python实现，无需安装服务器
3. 自动管理embedding（可选）
4. 适合快速原型开发和学习

官网：https://www.trychroma.com/
"""

import chromadb  # 导入ChromaDB
from sentence_transformers import SentenceTransformer  # 导入embedding模型


class ChromaDBDemo:
    """
    ChromaDB 封装类
    
    ChromaDB的核心概念：
    - Collection: 类似于表，存储一组向量
    - Document: 原始文本
    - Embedding: 文本的向量表示
    - Metadata: 附加信息（可选）
    - ID: 唯一标识
    """
    
    def __init__(self, collection_name: str = "rag_demo"):
        """
        初始化ChromaDB
        
        ChromaDB支持两种模式：
        1. 内存模式：数据存在内存中，程序结束就没了
        2. 持久化模式：数据保存到磁盘
        """
        # 创建ChromaDB客户端（内存模式）
        # 如果要持久化，用: chromadb.PersistentClient(path="./chroma_data")
        self.client = chromadb.Client()
        
        # 创建或获取一个Collection（类似数据库中的表）
        # 如果Collection已存在则获取，不存在则创建
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG演示用的向量集合"}  # 可选的元数据
        )
        
        # 加载embedding模型
        # paraphrase-multilingual-MiniLM-L12-v2 支持中文，维度384
        print("正在加载embedding模型...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("模型加载完成！")
    
    def add_documents(self, texts: list, metadatas: list = None):
        """
        添加文档到ChromaDB
        
        参数：
            texts: 文本列表，例如 ["苹果是水果", "香蕉是黄色的"]
            metadatas: 元数据列表（可选），例如 [{"category": "fruit"}, {"category": "fruit"}]
        """
        # 生成唯一ID列表
        # ChromaDB要求每个文档有唯一ID
        ids = [f"doc_{i}" for i in range(len(texts))]
        
        # 将文本转换为向量
        # encode() 方法会自动处理分词、token化等步骤
        embeddings = self.model.encode(texts).tolist()
        
        # 添加到Collection
        # ChromaDB会自动建立索引，加速后续搜索
        self.collection.add(
            ids=ids,                          # 文档ID
            embeddings=embeddings,            # 向量列表
            documents=texts,                  # 原始文本
            metadatas=metadatas               # 元数据（可选）
        )
        
        print(f"成功添加 {len(texts)} 条文档到ChromaDB")
    
    def search(self, query: str, top_k: int = 3) -> list:
        """
        搜索相似文档
        
        参数：
            query: 查询文本
            top_k: 返回前k个最相似的结果
        
        返回：
            [{"id": "...", "text": "...", "score": 0.95}, ...]
        """
        # 将查询文本转换为向量
        query_embedding = self.model.encode([query]).tolist()
        
        # 执行搜索
        # ChromaDB使用近似最近邻(ANN)算法，速度快
        results = self.collection.query(
            query_embeddings=query_embedding,  # 查询向量
            n_results=top_k,                   # 返回结果数量
            include=["documents", "distances", "metadatas"]  # 要返回的信息
        )
        
        # 整理结果格式
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "distance": results['distances'][0][i],  # 距离越小越相似
                "metadata": results['metadatas'][0][i] if results['metadatas'] else None
            })
        
        return formatted_results
    
    def count(self) -> int:
        """返回Collection中的文档数量"""
        return self.collection.count()
    
    def delete_all(self):
        """删除Collection中的所有文档"""
        # 获取所有ID
        all_data = self.collection.get()
        if all_data['ids']:
            self.collection.delete(ids=all_data['ids'])
        print("已清空所有文档")


# 测试代码
if __name__ == "__main__":
    print("=== ChromaDB 测试 ===\n")
    
    # 创建ChromaDB实例
    db = ChromaDBDemo("test_collection")
    
    # 测试数据
    documents = [
        "Python是一种流行的编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "自然语言处理处理文本数据",
        "向量数据库存储高维向量",
    ]
    
    metadatas = [
        {"category": "编程", "difficulty": "入门"},
        {"category": "AI", "difficulty": "中级"},
        {"category": "AI", "difficulty": "高级"},
        {"category": "NLP", "difficulty": "中级"},
        {"category": "数据库", "difficulty": "中级"},
    ]
    
    # 添加文档
    db.add_documents(documents, metadatas)
    
    # 搜索测试
    print("\n=== 搜索测试 ===")
    query = "什么是人工智能"
    print(f"查询: {query}")
    
    results = db.search(query, top_k=3)
    
    for i, result in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(f"  文本: {result['text']}")
        print(f"  距离: {result['distance']:.4f}")
        print(f"  元数据: {result['metadata']}")

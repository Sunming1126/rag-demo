"""
Milvus Lite 向量数据库示例

Milvus 特点：
1. 专业级向量数据库，支持分布式部署
2. 功能最丰富：过滤、混合查询等
3. 适合生产环境
4. Milvus Lite是轻量版，无需安装服务器

注意：Milvus Lite（pymilvus）适合学习和小项目
生产环境需要用完整的Milvus服务

官网：https://milvus.io/
"""

from pymilvus import (
    connections,    # 数据库连接
    Collection,     # 集合（类似表）
    FieldSchema,    # 字段定义
    CollectionSchema,  # 集合结构定义
    DataType,       # 数据类型
    utility         # 工具函数
)
from sentence_transformers import SentenceTransformer  # embedding模型


class MilvusLiteDemo:
    """
    Milvus Lite 封装类
    
    Milvus的核心概念：
    - Collection: 类似于数据库中的表
    - Schema: 定义数据结构（字段、类型）
    - Field: 字段，可以是向量、标量等
    - Index: 索引，加速搜索
    
    与其他数据库的区别：
    - ChromaDB: 简单易用，适合原型
    - FAISS: 纯向量检索，需要自己管理文本
    - Milvus: 功能完整，支持复杂查询
    """
    
    def __init__(self, collection_name: str = "rag_demo"):
        """
        初始化Milvus Lite
        
        Milvus Lite直接在本地创建数据库文件
        不需要启动服务器
        """
        self.collection_name = collection_name
        
        # 连接到Milvus Lite（本地文件模式）
        # uri指向本地SQLite文件
        connections.connect(
            alias="default",
            uri="./milvus_demo.db"  # 数据保存在这个文件中
        )
        
        # 加载embedding模型
        print("正在加载embedding模型...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("模型加载完成！")
        
        # 定义向量维度
        self.dimension = 384
        
        # 创建Collection（如果不存在）
        self._create_collection()
    
    def _create_collection(self):
        """
        创建Collection的Schema
        
        Milvus需要先定义数据结构，再使用
        这点和ChromaDB不同（ChromaDB自动处理）
        """
        # 检查Collection是否已存在
        if utility.has_collection(self.collection_name):
            # 如果存在，直接获取
            self.collection = Collection(self.collection_name)
            print(f"已加载现有Collection: {self.collection_name}")
            return
        
        # 定义字段
        # 1. 主键字段（自增ID）
        id_field = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,      # 设为主键
            auto_id=True          # 自动递增
        )
        
        # 2. 文本字段
        text_field = FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=2000       # 最大字符数
        )
        
        # 3. 向量字段
        vector_field = FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=self.dimension    # 向量维度
        )
        
        # 4. 元数据字段（可选）
        category_field = FieldSchema(
            name="category",
            dtype=DataType.VARCHAR,
            max_length=100
        )
        
        # 创建Schema
        schema = CollectionSchema(
            fields=[id_field, text_field, vector_field, category_field],
            description="RAG演示用的向量集合"
        )
        
        # 创建Collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        print(f"已创建新Collection: {self.collection_name}")
    
    def _create_index(self):
        """
        创建向量索引
        
        Milvus支持多种索引类型：
        - FLAT: 暴力搜索，精确
        - IVF_FLAT: 倒排索引
        - HNSW: 图索引，速度快
        - IVF_PQ: 乘积量化，省内存
        
        这里用HNSW，速度和精度都不错
        """
        # 检查是否已有索引
        if self.collection.has_index():
            return
        
        # 定义索引参数
        index_params = {
            "index_type": "HNSW",     # HNSW图索引
            "metric_type": "COSINE",  # 使用余弦相似度
            "params": {
                "M": 16,              # 每个节点的连接数
                "efConstruction": 200  # 构建时的搜索范围
            }
        }
        
        # 创建索引（在embedding字段上）
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        print("已创建HNSW索引")
    
    def add_documents(self, texts: list, categories: list = None):
        """
        添加文档到Milvus
        
        参数：
            texts: 文本列表
            categories: 分类列表（可选）
        """
        # 将文本转换为向量
        embeddings = self.model.encode(texts).tolist()
        
        # 准备数据（按照Schema中的字段顺序）
        data = [
            texts,        # text字段
            embeddings,   # embedding字段
            categories or ["unknown"] * len(texts)  # category字段
        ]
        
        # 插入数据
        mr = self.collection.insert(data)
        
        # 创建索引（如果还没创建）
        self._create_index()
        
        # 刷新数据，使其可搜索
        self.collection.flush()
        
        print(f"成功添加 {len(texts)} 条文档到Milvus")
    
    def search(self, query: str, top_k: int = 3, category_filter: str = None) -> list:
        """
        搜索相似文档
        
        Milvus的优势：支持过滤查询
        可以在搜索的同时按类别过滤
        """
        # 将查询转换为向量
        query_embedding = self.model.encode([query]).tolist()
        
        # 加载Collection到内存（搜索前必须）
        self.collection.load()
        
        # 定义搜索参数
        search_params = {
            "metric_type": "COSINE",  # 余弦相似度
            "params": {"ef": 100}     # HNSW搜索参数
        }
        
        # 构建过滤表达式（可选）
        expr = None
        if category_filter:
            expr = f'category == "{category_filter}"'
        
        # 执行搜索
        results = self.collection.search(
            data=query_embedding,        # 查询向量
            anns_field="embedding",      # 搜索的向量字段
            param=search_params,         # 搜索参数
            limit=top_k,                 # 返回数量
            expr=expr,                   # 过滤表达式
            output_fields=["text", "category"]  # 要返回的字段
        )
        
        # 整理结果
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "id": hit.id,
                "text": hit.entity.get("text"),
                "category": hit.entity.get("category"),
                "score": hit.score,  # 相似度得分（越高越相似）
                "distance": hit.distance  # 距离（越小越相似）
            })
        
        return formatted_results
    
    def count(self) -> int:
        """返回Collection中的文档数量"""
        return self.collection.num_entities
    
    def delete_all(self):
        """删除所有数据"""
        # Milvus删除数据需要用表达式
        self.collection.delete(expr="id >= 0")
        print("已清空所有文档")


# 测试代码
if __name__ == "__main__":
    print("=== Milvus Lite 测试 ===\n")
    
    # 创建Milvus实例
    db = MilvusLiteDemo("test_collection")
    
    # 测试数据
    documents = [
        "Python是一种流行的编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "自然语言处理处理文本数据",
        "向量数据库存储高维向量",
    ]
    
    categories = ["编程", "AI", "AI", "NLP", "数据库"]
    
    # 添加文档
    db.add_documents(documents, categories)
    
    # 搜索测试
    print("\n=== 搜索测试 ===")
    query = "什么是人工智能"
    print(f"查询: {query}")
    
    results = db.search(query, top_k=3)
    
    for i, result in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(f"  文本: {result['text']}")
        print(f"  分类: {result['category']}")
        print(f"  相似度得分: {result['score']:.4f}")
    
    # 带过滤的搜索
    print("\n=== 带过滤的搜索 ===")
    query = "什么是编程"
    print(f"查询: {query} (只搜索AI分类)")
    
    results = db.search(query, top_k=3, category_filter="AI")
    
    for i, result in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(f"  文本: {result['text']}")
        print(f"  分类: {result['category']}")

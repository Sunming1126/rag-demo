"""
Milvus Lite 向量数据库示例

Milvus 特点：
1. 专业级向量数据库，支持分布式部署
2. 功能丰富：过滤查询、混合搜索等
3. 适合生产环境
4. Milvus Lite 是轻量版，无需安装服务器

注意：新版 pymilvus (>=3.0) 使用 MilvusClient 替代旧的 ORM API

官网：https://milvus.io/
"""

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer


class MilvusLiteDemo:
    """
    Milvus Lite 封装类（使用新版 MilvusClient API）

    核心概念:
    - Collection: 类似数据库中的"表"
    - Schema: 定义字段结构（自动管理或手动定义）
    - Index: 加速搜索的索引（MilvusClient 自动管理）

    与 ChromaDB / FAISS 的区别:
    - ChromaDB: 纯 Python，适合原型
    - FAISS:   纯向量检索，需自己管理文本
    - Milvus:  完整数据库功能，支持过滤查询
    """

    def __init__(self, collection_name: str = "rag_demo"):
        """
        初始化 Milvus Lite

        Milvus Lite 直接用本地文件存储数据，无需启动服务器。
        新版推荐使用 MilvusClient（替代旧的 connections.connect）。
        """
        self.collection_name = collection_name

        # ── 创建 MilvusClient（本地文件模式）──
        # uri 指向 SQLite 数据库文件，Milvus Lite 用它持久化数据
        self.client = MilvusClient(uri="./milvus_demo.db")
        print(f"✅ 已连接 Milvus Lite (collection: {collection_name})")

        # ── 加载嵌入模型 ──
        print("正在加载 embedding 模型...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.dimension = 384  # 模型输出维度
        print("✅ 模型加载完成！")

        # ── 创建 Collection ──
        self._create_collection()

    def _create_collection(self):
        """
        创建 Collection（如果不存在）

        新版 MilvusClient 的 Schema 定义比旧版 ORM API 简洁很多。
        不需要手动管理 Collection 对象的 load/flush 状态。
        """
        # 如果已存在，直接返回
        if self.client.has_collection(self.collection_name):
            print(f"📂 已加载现有 Collection: {self.collection_name}")
            return

        # ── 定义 Schema ──
        # MilvusClient 使用字典格式定义 Schema（比旧版 FieldSchema 更简洁）
        schema = MilvusClient.create_schema(
            auto_id=True,          # ID 自动递增
            enable_dynamic_field=False,  # 不使用动态字段（字段严格定义）
        )

        # 添加字段: ID（主键，自动递增）
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # 添加字段: 文本内容
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)
        # 添加字段: 向量（384 维）
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        # 添加字段: 分类标签（可选，用于过滤查询）
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=100)

        # ── 创建 Collection ──
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        # ── 创建索引 ──
        # 索引参数:
        #   index_type: HNSW（图索引，速度和精度平衡好）
        #   metric_type: COSINE（余弦相似度）
        #   params.M: 每个节点的连接数
        #   params.efConstruction: 构建时的搜索范围
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )

        print(f"✅ 已创建新 Collection: {self.collection_name}")

    def add_documents(self, texts: list, categories: list = None):
        """
        添加文档到 Milvus

        参数:
            texts: 文本列表
            categories: 分类标签列表（可选）
        """
        if categories is None:
            categories = ["default"] * len(texts)

        # 将文本转为向量
        embeddings = self.model.encode(texts).tolist()

        # 准备数据（字典格式，符合 Schema）
        data = []
        for i, (text, embedding, category) in enumerate(zip(texts, embeddings, categories)):
            data.append({
                "text": text,
                "embedding": embedding,
                "category": category,
            })

        # 插入数据
        # MilvusClient.insert() 自动处理索引和持久化
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )

        print(f"✅ 成功添加 {len(texts)} 条文档到 Milvus (insert_count={result['insert_count']})")

    def search(self, query: str, top_k: int = 3, category_filter: str = None) -> list:
        """
        搜索相似文档

        Milvus 的优势在于支持过滤查询（类似 SQL 的 WHERE 子句），
        可以在搜索时按条件筛选数据。

        参数:
            query: 查询文本
            top_k: 返回前 k 条结果
            category_filter: 按分类过滤（可选）

        返回:
            [{"id": ..., "text": ..., "category": ..., "distance": ...}, ...]
        """
        # 搜索前先加载 Collection 到内存
        self.client.load_collection(self.collection_name)

        # 将查询转为向量
        query_embedding = self.model.encode([query]).tolist()

        # 构建过滤条件（可选）
        # MilvusClient 使用字符串表达式进行过滤，规则类似 SQL
        filter_expr = None
        if category_filter:
            filter_expr = f'category == "{category_filter}"'

        # 执行搜索
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_embedding,            # 查询向量（list of list）
            anns_field="embedding",           # 搜索的向量字段
            search_params={
                "metric_type": "COSINE",      # 余弦相似度
                "params": {"ef": 100},        # HNSW 搜索范围
            },
            limit=top_k,                      # 返回数量
            filter=filter_expr,               # 过滤表达式
            output_fields=["text", "category"],  # 需要返回的字段
        )

        # 整理结果
        # results 结构: [[{id, distance, entity}, ...], ...]
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "id": hit["id"],
                "text": hit["entity"].get("text", ""),
                "category": hit["entity"].get("category", ""),
                "distance": hit["distance"],  # COSINE 距离（越小越相似，范围 0-2）
            })

        return formatted_results

    def count(self) -> int:
        """返回 Collection 中的文档数量"""
        self.client.load_collection(self.collection_name)
        stats = self.client.query(
            collection_name=self.collection_name,
            output_fields=["count(*)"],
        )
        return stats[0]["count(*)"]

    def num_entities(self) -> int:
        """另一种获取文档数量的方式（使用 describe_collection）"""
        info = self.client.describe_collection(self.collection_name)
        return info.get("num_entities", 0)

    def delete_all(self):
        """删除所有数据（删除并重建 Collection）"""
        self.client.drop_collection(collection_name=self.collection_name)
        self._create_collection()
        print("🗑️ 已清空所有文档")


# ─── 测试代码 ───
if __name__ == "__main__":
    print("=" * 60)
    print("🧪  Milvus Lite 测试（新版 MilvusClient API）")
    print("=" * 60)

    # 创建实例
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
    print("\n" + "-" * 40)
    print("📥  添加文档")
    print("-" * 40)
    db.add_documents(documents, categories)
    print(f"   当前文档数: {db.count()}")

    # 搜索测试
    print("\n" + "-" * 40)
    print("🔍  搜索测试")
    print("-" * 40)
    query = "什么是人工智能"
    print(f"   查询: 「{query}」")

    results = db.search(query, top_k=3)
    for i, r in enumerate(results):
        print(f"\n  结果 {i+1}:")
        print(f"    文本:   {r['text']}")
        print(f"    分类:   {r['category']}")
        print(f"    距离:   {r['distance']:.4f}")

    # 带过滤的搜索
    print("\n" + "-" * 40)
    print("🔍  带过滤搜索 (category='AI')")
    print("-" * 40)
    query = "什么是编程"
    print(f"   查询: 「{query}」 (仅搜索 AI 分类)")

    results = db.search(query, top_k=3, category_filter="AI")
    for i, r in enumerate(results):
        print(f"\n  结果 {i+1}:")
        print(f"    文本:   {r['text']}")
        print(f"    分类:   {r['category']}")
        print(f"    距离:   {r['distance']:.4f}")

    # 清理
    print("\n" + "-" * 40)
    print("🧹  清理")
    print("-" * 40)
    db.delete_all()
    print("✅ 测试完成！")

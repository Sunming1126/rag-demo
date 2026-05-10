"""
FAISS 向量数据库示例

FAISS 特点：
1. Facebook开源的高性能向量检索库
2. 用C++实现，速度极快
3. 支持GPU加速
4. 适合大规模向量检索（百万级、十亿级）

注意：FAISS不是完整的数据库，只是向量检索库
需要自己管理文本存储和元数据

官网：https://github.com/facebookresearch/faiss
"""

import faiss  # 导入FAISS
import numpy as np  # 导入numpy，FAISS需要numpy数组
from sentence_transformers import SentenceTransformer  # 导入embedding模型


class FAISSDemo:
    """
    FAISS 封装类
    
    FAISS的核心概念：
    - Index: 索引结构，决定如何存储和搜索向量
    - Flat: 暴力搜索，精确但慢
    - IVF: 倒排索引，速度和精度的平衡
    - HNSW: 图索引，速度快，内存占用大
    
    常用索引类型：
    - IndexFlatL2: 暴力搜索，L2距离（精确）
    - IndexFlatIP: 暴力搜索，内积（用于余弦相似度）
    - IndexIVFFlat: 倒排索引，速度快
    - IndexHNSWFlat: HNSW图索引，速度最快
    """
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """
        初始化FAISS
        
        参数：
            dimension: 向量维度，取决于embedding模型
                      MiniLM模型输出384维
            index_type: 索引类型
                       "flat" = 暴力搜索（精确，适合小数据集）
                       "ivf" = 倒排索引（速度快，适合大数据集）
                       "hnsw" = HNSW图索引（速度最快）
        """
        self.dimension = dimension  # 保存向量维度
        
        # 根据索引类型创建不同的索引
        if index_type == "flat":
            # Flat索引：暴力搜索，计算所有向量的距离
            # 优点：精确，无误差
            # 缺点：数据量大时慢
            # L2 = 欧几里得距离（也可以用IP内积）
            self.index = faiss.IndexFlatL2(dimension)
            
        elif index_type == "ivf":
            # IVF索引：先聚类，搜索时只在相关聚类中搜索
            # 需要先训练（train）才能使用
            nlist = 100  # 聚类数量
            quantizer = faiss.IndexFlatL2(dimension)  # 量化器
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
        elif index_type == "hnsw":
            # HNSW索引：基于图的索引
            # 优点：搜索速度快
            # 缺点：内存占用大
            M = 32  # 每个节点的连接数
            self.index = faiss.IndexHNSWFlat(dimension, M)
        
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
        
        # FAISS不存储原始文本，需要自己维护一个列表
        self.texts = []      # 存储原始文本
        self.is_trained = False  # 标记索引是否已训练
        
        # 加载embedding模型
        print("正在加载embedding模型...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("模型加载完成！")
    
    def add_documents(self, texts: list):
        """
        添加文档到FAISS
        
        注意：FAISS只存储向量，不存储文本
        所以我们需要同时维护texts列表
        """
        # 将文本转换为向量
        embeddings = self.model.encode(texts)
        
        # FAISS要求输入是float32类型的numpy数组
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # 检查索引是否需要训练
        # IVF索引需要先训练才能添加数据
        if not self.is_trained:
            if hasattr(self.index, 'train'):
                # 训练索引（对于Flat索引，train是空操作）
                self.index.train(embeddings)
            self.is_trained = True
        
        # 添加向量到索引
        self.index.add(embeddings)
        
        # 同时保存原始文本
        self.texts.extend(texts)
        
        print(f"成功添加 {len(texts)} 条文档到FAISS")
    
    def search(self, query: str, top_k: int = 3) -> list:
        """
        搜索相似文档
        
        FAISS搜索返回：
        - D: 距离矩阵（越小越相似）
        - I: 索引矩阵（对应的文档编号）
        """
        # 将查询文本转换为向量
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # 执行搜索
        # search()返回两个数组：距离和索引
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 整理结果
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]  # 文档索引
            dist = distances[0][i]  # 距离
            
            # 检查索引是否有效（-1表示无效）
            if idx != -1:
                results.append({
                    "id": idx,
                    "text": self.texts[idx],
                    "distance": float(dist)  # L2距离，越小越相似
                })
        
        return results
    
    def count(self) -> int:
        """返回索引中的向量数量"""
        return self.index.ntotal
    
    def save(self, filepath: str):
        """
        保存索引到文件
        
        FAISS索引可以序列化，方便下次加载
        """
        faiss.write_index(self.index, filepath)
        print(f"索引已保存到: {filepath}")
    
    def load(self, filepath: str):
        """从文件加载索引"""
        self.index = faiss.read_index(filepath)
        print(f"索引已从 {filepath} 加载")


# 测试代码
if __name__ == "__main__":
    print("=== FAISS 测试 ===\n")
    
    # 创建FAISS实例（使用Flat暴力索引）
    db = FAISSDemo(dimension=384, index_type="flat")
    
    # 测试数据
    documents = [
        "Python是一种流行的编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "自然语言处理处理文本数据",
        "向量数据库存储高维向量",
    ]
    
    # 添加文档
    db.add_documents(documents)
    
    # 搜索测试
    print("\n=== 搜索测试 ===")
    query = "什么是人工智能"
    print(f"查询: {query}")
    
    results = db.search(query, top_k=3)
    
    for i, result in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(f"  索引ID: {result['id']}")
        print(f"  文本: {result['text']}")
        print(f"  距离: {result['distance']:.4f}")
    
    print(f"\n索引中向量数量: {db.count()}")

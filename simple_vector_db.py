"""
最简单的向量数据库实现
目的：理解向量数据库的底层原理

核心概念：
1. 向量数据库 = 存储向量 + 搜索相似向量
2. 相似度计算 = 余弦相似度（衡量两个向量的方向是否一致）
3. 索引 = 快速找到最相似的向量
"""

import numpy as np  # 导入numpy，用于数学计算
from typing import List, Tuple  # 导入类型提示，让代码更清晰


class SimpleVectorDB:
    """
    最简单的向量数据库实现
    
    工作原理：
    1. 把文本转换成向量（一串数字）
    2. 存储这些向量
    3. 搜索时，计算查询向量和所有存储向量的距离
    4. 返回距离最近的向量（最相似的）
    """
    
    def __init__(self):
        # 初始化：创建空的向量存储列表
        self.vectors = []      # 存储所有向量
        self.texts = []        # 存储对应的原始文本
        self.ids = []          # 存储每个向量的ID
        self.next_id = 0       # 自增ID计数器
    
    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        余弦相似度原理：
        - 把向量想象成从原点出发的箭头
        - 余弦相似度衡量两个箭头的方向是否一致
        - 值范围：-1到1
        - 1 = 方向完全一致（最相似）
        - 0 = 方向垂直（无关）
        - -1 = 方向完全相反
        
        公式：cos(θ) = (A·B) / (|A| * |B|)
        """
        # 计算点积（A·B）
        dot_product = np.dot(vec_a, vec_b)
        
        # 计算向量A的长度（模）
        norm_a = np.linalg.norm(vec_a)
        
        # 计算向量B的长度（模）
        norm_b = np.linalg.norm(vec_b)
        
        # 避免除以0的错误
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # 返回余弦相似度
        return dot_product / (norm_a * norm_b)
    
    def add(self, text: str, embedding: np.ndarray) -> int:
        """
        添加一条数据到数据库
        
        参数：
            text: 原始文本
            embedding: 文本对应的向量（从embedding模型获取）
        
        返回：
            分配的ID
        """
        # 生成新ID
        new_id = self.next_id
        
        # 存储向量
        self.vectors.append(embedding)
        
        # 存储原始文本
        self.texts.append(text)
        
        # 存储ID
        self.ids.append(new_id)
        
        # ID计数器加1
        self.next_id += 1
        
        return new_id
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[int, str, float]]:
        """
        搜索最相似的向量
        
        参数：
            query_embedding: 查询向量
            top_k: 返回前k个最相似的结果
        
        返回：
            [(id, text, similarity_score), ...]
        """
        # 如果数据库为空，返回空列表
        if len(self.vectors) == 0:
            return []
        
        # 计算查询向量和所有存储向量的相似度
        similarities = []
        for i, stored_vector in enumerate(self.vectors):
            # 计算余弦相似度
            sim = self.cosine_similarity(query_embedding, stored_vector)
            similarities.append((self.ids[i], self.texts[i], sim))
        
        # 按相似度降序排序（最相似的在前面）
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # 返回前top_k个结果
        return similarities[:top_k]
    
    def delete(self, target_id: int) -> bool:
        """
        删除指定ID的数据
        
        参数：
            target_id: 要删除的ID
        
        返回：
            是否删除成功
        """
        # 查找ID对应的索引
        if target_id in self.ids:
            index = self.ids.index(target_id)
            
            # 删除对应位置的数据
            self.vectors.pop(index)
            self.texts.pop(index)
            self.ids.pop(index)
            
            return True
        
        return False
    
    def count(self) -> int:
        """返回数据库中存储的向量数量"""
        return len(self.vectors)
    
    def display(self):
        """显示数据库中的所有数据"""
        print(f"\n=== 简易向量数据库内容 (共{self.count()}条) ===")
        for i in range(len(self.ids)):
            print(f"ID: {self.ids[i]}")
            print(f"文本: {self.texts[i]}")
            print(f"向量维度: {len(self.vectors[i])}")
            print("-" * 40)


# 测试代码
if __name__ == "__main__":
    # 这里只测试数据库功能，不使用embedding模型
    # 用随机向量模拟embedding
    
    print("=== 测试简易向量数据库 ===\n")
    
    # 创建数据库实例
    db = SimpleVectorDB()
    
    # 添加一些测试数据（用随机向量模拟）
    test_data = [
        ("苹果是一种水果", np.random.rand(384)),  # 384维向量
        ("香蕉是黄色的", np.random.rand(384)),
        ("今天天气很好", np.random.rand(384)),
    ]
    
    for text, vector in test_data:
        db.add(text, vector)
    
    # 显示数据库内容
    db.display()
    
    # 测试搜索（用一个随机查询向量）
    query_vector = np.random.rand(384)
    results = db.search(query_vector, top_k=2)
    
    print("\n=== 搜索结果 ===")
    for id, text, score in results:
        print(f"ID: {id}, 相似度: {score:.4f}, 文本: {text}")

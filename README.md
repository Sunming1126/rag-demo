# RAG 学习 Demo

一个帮助理解向量数据库和RAG原理的学习项目。

## 📁 项目结构

```
rag-demo/
├── simple_vector_db.py  # 自己实现的简易向量数据库（理解原理）
├── chromadb_demo.py     # ChromaDB 示例（最简单易用）
├── faiss_demo.py        # FAISS 示例（性能最高）
├── milvus_demo.py       # Milvus Lite 示例（功能最全）
├── rag_main.py          # 主程序，整合所有数据库的RAG演示
├── requirements.txt     # 依赖包列表
└── README.md            # 说明文档（本文件）
```

## 🎯 学习目标

1. **理解向量数据库的原理**：看 `simple_vector_db.py`，了解底层实现
2. **对比不同数据库**：看 chromadb、faiss、milvus 三个示例
3. **理解RAG流程**：运行 `rag_main.py`，看完整的检索增强生成流程

## 📦 安装依赖

```bash
# 进入项目目录
cd ~/Desktop/rag-demo

# 创建conda环境（推荐）
conda create -n rag-demo python=3.10
conda activate rag-demo

# 安装依赖
pip install -r requirements.txt
```

## 🚀 运行演示

### 运行完整RAG演示
```bash
python rag_main.py
```

### 单独测试每个数据库

```bash
# 测试简易向量数据库
python simple_vector_db.py

# 测试 ChromaDB
python chromadb_demo.py

# 测试 FAISS
python faiss_demo.py

# 测试 Milvus Lite
python milvus_demo.py
```

## 📚 向量数据库对比

| 特性 | ChromaDB | FAISS | Milvus Lite |
|------|----------|-------|-------------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 性能 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 功能丰富度 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 适用场景 | 原型开发 | 大规模检索 | 生产环境 |
| 需要服务器 | 否 | 否 | Lite版不需要 |

### 各数据库特点

**ChromaDB**
- 最简单易用，几行代码就能用
- 自动处理embedding转换
- 适合快速原型开发和学习

**FAISS**
- Facebook开源，C++实现，性能极高
- 支持GPU加速
- 只做向量检索，需要自己管理文本

**Milvus Lite**
- 功能最完整，支持过滤查询
- 需要先定义Schema（数据结构）
- 适合生产环境，支持分布式部署

## 🔍 核心概念解释

### 1. 向量（Vector）
```
文本: "苹果是一种水果"
     ↓ 转换
向量: [0.23, -0.45, 0.67, ..., 0.12]  (384个数字)
```

### 2. 相似度计算
```
向量A: [0.23, -0.45, 0.67]
向量B: [0.25, -0.43, 0.65]
        ↓ 计算
相似度: 0.98 (越接近1越相似)
```

### 3. RAG工作流程
```
用户提问: "什么是深度学习？"
     ↓
1. 检索: 在向量数据库中找到最相关的文档
     ↓
2. 增强: 把相关文档和问题拼接在一起
     ↓
3. 生成: 发送给大模型生成答案
```

## 💡 学习建议

1. **先看原理**：仔细阅读 `simple_vector_db.py` 的注释
2. **动手修改**：尝试修改参数，观察结果变化
3. **添加数据**：用自己的文本测试搜索效果
4. **对比性能**：运行 `rag_main.py` 看速度对比

## 🤔 常见问题

**Q: 为什么需要向量数据库？**
A: 传统数据库搜索精确匹配，向量数据库搜索语义相似。比如搜"苹果"，能匹配到"一种水果"。

**Q: 什么是Embedding？**
A: 把文本转换成向量的过程。相似文本转换后的向量在空间中距离近。

**Q: 选择哪个数据库？**
A: 
- 学习和原型 → ChromaDB
- 性能要求高 → FAISS
- 生产环境 → Milvus

## 📖 扩展学习

- [ChromaDB 官方文档](https://docs.trychroma.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Milvus 官方文档](https://milvus.io/docs)
- [Sentence-Transformers](https://www.sbert.net/)

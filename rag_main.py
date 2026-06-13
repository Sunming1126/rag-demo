"""
🦜 RAG (检索增强生成) 学习演示 - 主程序

├── 📂 步骤 1: 数据加载  (MarkdownLoader → data_loader.py)
├── ✂️  步骤 2: 文本切分  (RecursiveCharacterTextSplitter → text_splitter.py)
├── 🔢 步骤 3: 向量嵌入  (SentenceTransformer → 384维向量)
├── 💾 步骤 4: 向量存储  (SimpleVectorDB / ChromaDB / FAISS / Milvus)
├── 🔍 步骤 5: 相似度检索 (余弦相似度 / L2距离)
└── 💬 步骤 6: 生成回答  (结合检索结果 + LLM)

运行:
    python rag_main.py

学习建议:
    1. 先运行一次，观察输出
    2. 修改 chunk_size 和 chunk_overlap，看检索效果变化
    3. 替换 data/ 目录下的 .md 文件，使用自己的知识库
    4. 对比不同向量数据库的检索效果
"""

import sys
import os

# 确保可以导入当前目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import MarkdownLoader
from text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  步骤 2: 文本切分演示
# ═══════════════════════════════════════════════════════════════

def step2_split_documents(documents):
    """
    ✂️  文本切分 (Chunking) — RAG 流程中最关键的环节

    用递归字符分割器将长文档切分为适合检索的文本块。

    为什么 chunk_size 选 512?
        - 对中文来说 512 字符 ≈ 200-300 个有意义的中文词
        - 嵌入模型的输入限制通常是 512 tokens (约 700+ 中文字符)
        - 这个大小刚好容纳 1-2 个 QA 对

    为什么 overlap 选 128?
        - 128/512 = 25% 的重叠率
        - 假设一段文本在「。」处被切断:
          "...A。|B..." → 块1 以 A。结尾，块2 以 B... 开头
          → 查询「A 的影响」可能找不到块2
        - 有了 overlap:
          "...A。|B..." → 块1 以 A。结尾，块2 以 。B... 开头
          → 两个块都包含 A 的信息
    """
    print("\n" + "=" * 60)
    print("✂️  步骤 2: 文本切分 (Text Chunking)")
    print("=" * 60)

    # ── 创建递归分割器 ──
    # 这两个参数对检索效果影响最大，建议多尝试不同组合:
    #   chunk_size:  [256, 512, 1024]
    #   chunk_overlap: [0, 64, 128, 256]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
    )

    print(f"\n📐  分割器配置:")
    print(f"     chunk_size    = {splitter.chunk_size} 字符")
    print(f"     chunk_overlap = {splitter.chunk_overlap} 字符")
    print(f"     分隔符优先级  :")
    for i, sep in enumerate(splitter.separators):
        display = repr(sep) if sep else "'' (字符级 fallback)"
        print(f"       {i+1:2d}. {display}")

    # ── 对每个文档进行切分 ──
    all_chunks = []  # [{"source": "xxx.md", "chunks": ["...", "..."]}, ...]

    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        all_chunks.append({
            "source": doc["filename"],
            "chunks": chunks,
        })

        # 打印切分统计
        print(f"\n📄  [{doc['filename']}]")
        print(f"    原始: {doc['length']:>6,} 字符 → 切分为 {len(chunks):>3} 个块")

        # 如果块数 > 0，显示前 3 个块的预览
        if chunks:
            n_preview = min(3, len(chunks))
            print(f"    前 {n_preview} 个块预览:")
            for i, chunk in enumerate(chunks[:n_preview]):
                preview = chunk[:120].replace("\n", "↵ ")
                if len(chunk) > 120:
                    preview += "..."
                print(f"      ── 块 #{i+1} (len={len(chunk):>3}) ── {preview}")

        if len(chunks) > 3:
            print(f"      ... 还有 {len(chunks) - 3} 个块未显示")

    return all_chunks


# ═══════════════════════════════════════════════════════════════
#  步骤 3-5: 嵌入 + 存储 + 检索
# ═══════════════════════════════════════════════════════════════

def step3_5_embed_and_search(all_chunks, model):
    """
    嵌入式 → 向量存储 → 相似度检索 (三位一体演示)

    这是 RAG 的「检索」核心:
        1. 每个文本块 → 384 维向量 (嵌入)
        2. 向量存入数据库
        3. 用户查询 → 同样转为向量 → 搜索最近邻
    """
    print("\n" + "=" * 60)
    print("🔢  步骤 3-5: 嵌入 → 存储 → 检索")
    print("=" * 60)

    # ── 展平所有块 ──
    texts = []
    sources = []
    for doc_chunks in all_chunks:
        for chunk in doc_chunks["chunks"]:
            texts.append(chunk)
            sources.append(doc_chunks["source"])

    print(f"\n📊  共 {len(texts)} 个文本块")

    # ── 步骤 3: 生成嵌入向量 ──
    # 嵌入模型: paraphrase-multilingual-MiniLM-L12-v2
    #   - 多语言模型，支持中英文
    #   - 输出 384 维向量
    #   - MiniLM 架构，轻量快速
    print(f"\n⏳  生成嵌入向量...")
    print(f"     模型: {model._first_module().auto_model.config._name_or_path}")
    print(f"     输出维度: {model.get_embedding_dimension()}")

    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"   ✅ 嵌入完成! 形状: {embeddings.shape}")

    # ── 步骤 4: 存入 SimpleVectorDB ──
    # 使用教学级的 SimpleVectorDB (简单的余弦相似度 + 暴力搜索)
    # 也可以换成 ChromaDB / FAISS / Milvus (见其他 demo 文件)
    from simple_vector_db import SimpleVectorDB

    print(f"\n💾  存储到 SimpleVectorDB ...")
    vector_db = SimpleVectorDB()
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        vector_db.add(text, embedding)
    print(f"   ✅ 已存储 {vector_db.count()} 个向量")

    # ── 步骤 5: 检索演示 ──
    print(f"\n{'=' * 60}")
    print(f"🔍  步骤 5: 相似度检索演示")
    print(f"{'=' * 60}")

    # 准备几个测试查询
    test_queries = [
        "工作流和对话流有什么区别？",
        "如何提升提示词质量？",
        "怎么减少AI回答的幻觉？",
        "向量数据库有哪些类型？",
        "什么是RAG？",
    ]

    for query in test_queries:
        # 将查询转为向量
        query_vec = model.encode(query)

        # 在向量数据库中搜索 top-3 最相似的块
        results = vector_db.search(query_vec, top_k=3)

        print(f"\n{'─' * 55}")
        print(f"  ❓ 查询: 「{query}」")
        print(f"{'─' * 55}")

        for rank, (doc_id, text, score) in enumerate(results, 1):
            # 截断过长的文本
            display_text = text[:250].replace("\n", "↵ ")
            if len(text) > 250:
                display_text += "..."

            print(f"  #{rank}  [来自 {sources[doc_id]}]  相似度: {score:.4f}")
            print(f"     {display_text}")
            print()

    return vector_db, texts, sources


# ═══════════════════════════════════════════════════════════════
#  [可选] 步骤 6: 简单生成回答
# ═══════════════════════════════════════════════════════════════

def step6_generate(vector_db, texts, sources, model, query: str, top_k: int = 3):
    """
    💬  步骤 6: 生成回答 (Generation)

    这是完整的 RAG 最后一步:
        检索到的文本块 → 拼成上下文 → 交给 LLM → 生成答案

    注意: 这里用简单的「拼接 + 规则生成」模拟 LLM 输出。
    实际生产环境中会调用真正的 LLM API (如 Claude、GPT 等)。
    """
    print(f"\n{'=' * 60}")
    print(f"💬  步骤 6: 生成回答 (Generation)")
    print(f"{'=' * 60}")

    # 1. 检索
    query_vec = model.encode(query)
    results = vector_db.search(query_vec, top_k=top_k)

    # 2. 构建上下文
    context_parts = []
    for rank, (doc_id, text, score) in enumerate(results, 1):
        context_parts.append(f"[参考 {rank}] {text}")
    context = "\n\n".join(context_parts)

    print(f"\n  ❓ 查询: {query}")
    print(f"\n  📋 检索到的上下文 ({len(results)} 条):")
    for rank, (doc_id, text, score) in enumerate(results, 1):
        preview = text[:100].replace("\n", " ")
        print(f"     [{rank}] (score={score:.4f}) {preview}...")

    # 3. 模拟生成回答
    #    在实际 RAG 系统中，这里会调用 LLM:
    #       response = llm.chat(f"基于以下信息回答问题:\n{context}\n\n问题: {query}")
    response = _simulate_answer(query, context)

    print(f"\n  🤖 生成回答:")
    print(f"     {response}")
    print(f"\n  ℹ️  (以上是模拟回答。实际 RAG 会调用 LLM 基于检索到的上下文生成。)")

    return response


def _simulate_answer(query: str, context: str) -> str:
    """
    模拟 LLM 生成回答

    在一个真实的 RAG 系统中，这里会调用:
        anthropic.Anthropic().messages.create(...)   # Claude
        openai.ChatCompletion.create(...)             # GPT
        或其他 LLM API

    这里的实现只是从上下文中提取关键信息进行展示。
    """
    # 简单地从前 200 个字符中提取回答
    snippet = context[:200].strip()
    return f"根据检索到的资料:\n「{snippet}」\n\n实际使用时，可以将这些检索到的上下文拼接成 prompt，调用 LLM 生成最终回答。"


# ═══════════════════════════════════════════════════════════════
#  扩展实验: 不同 chunk 参数对比
# ═══════════════════════════════════════════════════════════════

def experiment_chunk_size(documents):
    """
    🧪 实验: 不同 chunk_size 对检索效果的影响

    尝试不同的参数组合，观察:
    - 小 chunk (256):  精确定位，但可能缺失上下文
    - 大 chunk (1024): 上下文完整，但检索噪音多
    """
    print("\n\n" + "=" * 60)
    print("🧪  实验: 不同 chunk_size 对比")
    print("=" * 60)

    configs = [
        {"chunk_size": 256, "chunk_overlap": 64,  "desc": "小块 + 小重叠"},
        {"chunk_size": 512, "chunk_overlap": 128, "desc": "中块 + 中等重叠 ✓ 推荐"},
        {"chunk_size": 1024, "chunk_overlap": 128, "desc": "大块 + 固定重叠"},
    ]

    all_text = documents[0]["content"] if documents else ""

    print(f"\n📄 使用文档: {documents[0]['filename'] if documents else 'N/A'}")
    print(f"   原文长度: {len(all_text):,} 字符\n")

    for cfg in configs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        chunks = splitter.split_text(all_text)
        sizes = [len(c) for c in chunks]

        print(f"  {cfg['desc']}:")
        print(f"    chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}")
        print(f"    → {len(chunks):>3} 个块, 平均 {sum(sizes)/len(sizes):.0f} 字符/块")
        print()


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    """
    完整 RAG 流程

    执行顺序:
        📂 加载 → ✂️ 切分 → 🔢 嵌入 → 💾 存储 → 🔍 检索 → 💬 生成
    """
    print("=" * 60)
    print("🦜  RAG (检索增强生成) 学习演示")
    print("     Retrieval-Augmented Generation Demo")
    print("=" * 60)

    # ════════════════════════════════════════════
    #  步骤 1: 数据加载
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📂  步骤 1: 加载 data/ 目录下的 .md 文件")
    print("=" * 60)

    loader = MarkdownLoader(data_dir="data")
    documents = loader.load_all()
    MarkdownLoader.print_summary(documents)

    if not documents:
        print("\n❌  没有找到文档。请将 .md 文件放入 data/ 目录后重试。")
        print("    data/ 目录已存在，可以直接放入您的知识库文件。")
        return

    # ════════════════════════════════════════════
    #  步骤 2: 文本切分
    # ════════════════════════════════════════════
    all_chunks = step2_split_documents(documents)

    # ════════════════════════════════════════════
    #  加载嵌入模型 (步骤 3-5 共用)
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🤖  加载嵌入模型")
    print("=" * 60)
    print("""
    模型: paraphrase-multilingual-MiniLM-L12-v2
     - 多语言支持 (含中文)
     - 输出 384 维向量
     - MiniLM 架构，速度快
""")

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # ════════════════════════════════════════════
    #  步骤 3-5: 嵌入 → 存储 → 检索
    # ════════════════════════════════════════════
    vector_db, texts, sources = step3_5_embed_and_search(all_chunks, model)

    # ════════════════════════════════════════════
    #  步骤 6: 生成回答 (可选)
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("💬  步骤 6: 基于检索结果生成回答")
    print("=" * 60)

    step6_generate(vector_db, texts, sources, model, "工作流和对话流有什么本质区别？")

    # ════════════════════════════════════════════
    #  实验: chunk 参数对比
    # ════════════════════════════════════════════
    experiment_chunk_size(documents)

    # ════════════════════════════════════════════
    #  总结
    # ════════════════════════════════════════════
    print("=" * 60)
    print("✅  RAG 演示完成!")
    print("=" * 60)
    print("""
📚 学习路径建议:

  1. 先理解「切分」— 看 text_splitter.py 的代码和注释
  2. 尝试修改 chunk_size / chunk_overlap 观察效果
  3. 将 data/ 目录下的 .md 文件替换成自己的知识库
  4. 学习其他向量数据库:
     - chromadb_demo.py  — ChromaDB (纯 Python, 入门友好)
     - faiss_demo.py     — FAISS (高性能, 适合大规模)
     - milvus_demo.py    — Milvus (分布式, 生产级)
  5. 接入真实 LLM (如 Claude API) 替代模拟回答

🔧 参数调优建议:
  - 技术文档/QA:    chunk_size=512, overlap=128
  - 长篇文章:        chunk_size=1024, overlap=256
  - 代码文档:         chunk_size=256, overlap=64
  - 对话/短文本:      chunk_size=256, overlap=0-32

📂 项目文件说明:
  - data_loader.py    — 从 data/ 读取 .md 文件
  - text_splitter.py  — 递归字符分割器 (核心学习文件)
  - simple_vector_db.py — 手写向量数据库 (教学用)
  - chromadb_demo.py    — ChromaDB 示例
  - faiss_demo.py       — FAISS 示例
  - milvus_demo.py      — Milvus Lite 示例
""")


if __name__ == "__main__":
    main()

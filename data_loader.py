"""
📂 数据加载器 - MarkdownLoader

功能: 从 data/ 目录读取所有 .md 文件，返回结构化内容

这是 RAG 流程的第一步：加载原始文档。
"""

import os
from pathlib import Path


class MarkdownLoader:
    """
    Markdown 文件加载器

    读取指定目录下的所有 .md 文件，返回包含文件名、标题、内容等信息的字典列表。

    用法:
        loader = MarkdownLoader(data_dir="data")
        docs = loader.load_all()
        loader.print_summary(docs)
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_all(self) -> list[dict]:
        """
        加载 data/ 目录下所有 .md 文件

        返回:
            [{"filename": 文件名, "title": 标题, "content": 全文内容, ...}, ...]
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"❌ 数据目录不存在: {self.data_dir}")

        documents = []
        md_files = sorted(self.data_dir.glob("*.md"))

        if not md_files:
            print(f"⚠️  在 {self.data_dir}/ 目录下没有找到 .md 文件")
            return documents

        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 从文件名提取标题（去掉 .md 后缀）
            title = md_file.stem

            documents.append({
                "filename": md_file.name,
                "title": title,
                "path": str(md_file),
                "content": content,
                "length": len(content),
                "line_count": content.count("\n") + 1,
            })

        return documents

    @staticmethod
    def print_summary(documents: list[dict]):
        """打印文档加载摘要"""
        print("=" * 60)
        print("📂  文档加载摘要")
        print("=" * 60)
        for doc in documents:
            print(f"  📁 {doc['filename']}")
            print(f"     标题: {doc['title']}")
            print(f"     大小: {doc['length']:,} 字符")
            print(f"     行数: {doc['line_count']:,} 行")
        print(f"\n  ✅ 共加载 {len(documents)} 个文件")


# ========== 独立运行测试 ==========
if __name__ == "__main__":
    loader = MarkdownLoader()
    docs = loader.load_all()
    MarkdownLoader.print_summary(docs)

    if docs:
        print(f"\n📝 第一个文档前 200 个字符预览:")
        print("-" * 40)
        print(docs[0]["content"][:200])

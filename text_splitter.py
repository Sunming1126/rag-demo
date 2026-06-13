"""
✂️  递归字符文本分割器 (Recursive Character Text Splitter)

RAG 流程中最关键的一步：将长文档切分为适合检索的「文本块」(chunks)。

┌─────────────────────────────────────────────────┐
│  为什么要切分 (Chunking)?                        │
│                                                  │
│  1. 嵌入模型有最大输入限制 (通常 512 tokens)      │
│  2. 细粒度的块 → 更精确的语义检索                 │
│  3. 过长的上下文会稀释 LLM 的注意力 (Lost in      │
│     the Middle 问题)                             │
│  4. 多个块可以覆盖文档的不同部分                   │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  为什么用「递归分割」?                           │
│                                                  │
│  从粗到细使用多级分隔符：                          │
│  Markdown标题 → 段落 → 句子 → 短语 → 字符        │
│                                                  │
│  优先在「语义边界」切分，保持块内语义完整。         │
│  如果粗粒度切分后块仍然太大，自动降级到下一级。    │
└─────────────────────────────────────────────────┘

参数选择:
    chunk_size=512    每个块的最大字符数
                        → 适合中文（约200-300个词）
                        → 太小：丢失上下文
                        → 太大：检索噪音多

    chunk_overlap=128  相邻块之间的重叠字符数
                        → 约为 chunk_size 的 25%
                        → 确保边界信息不被切断
                        → 例如 "...A|B..." 中 A 的结尾在 B 的开头重复
"""


class RecursiveCharacterTextSplitter:
    """
    递归字符文本分割器

    工作原理 (3 步):
        1. SPLIT:  用当前分隔符分割文本
        2. RECURSE: 对过大的子块，用更细的分隔符继续递归
        3. MERGE:   把小片段合并成块，相邻块间插入 overlap

    参数:
        chunk_size:     每个块的最大字符数 (默认: 512)
        chunk_overlap:  相邻块之间的重叠字符数 (默认: 128)
        separators:     分隔符列表，从粗粒度到细粒度
                        (默认值针对中英文 Markdown 优化)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        separators: list = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # ========== 分隔符设计思路 ==========
        # 从「语义最强」到「语义最弱」排列：
        #
        #   "\n## "    → Markdown H2 标题 (章节边界)
        #   "\n### "   → Markdown H3 标题 (子章节边界)
        #   "\n#### "  → Markdown H4 标题
        #   "\n\n"     → 段落边界
        #   "\n"       → 行边界
        #   "。"       → 中文句号 (句子边界)
        #   "！？"     → 中文句末标点
        #   "，"       → 中文逗号 (子句边界)
        #   "；"       → 中文分号
        #   " "        → 空格 (英文单词边界)
        #   ""         → 字符级别 (最终 fallback)
        #
        # 这样设计保证了：
        #   ✅ 优先在标题处切分 → 块内有完整主题
        #   ✅ 其次在段落处切分 → 块内有完整段落
        #   ✅ 再次在句子处切分 → 块内有完整句子
        #   ❌ 只有万不得已才在字符处切分
        self.separators = separators or [
            "\n## ",       # 1. Markdown H2
            "\n### ",      # 2. Markdown H3
            "\n#### ",     # 3. Markdown H4
            "\n\n",        # 4. 段落
            "\n",          # 5. 行
            "。",          # 6. 句号
            "！",          # 7. 感叹号
            "？",          # 8. 问号
            "，",          # 9. 逗号
            "；",          # 10. 分号
            " ",           # 11. 空格
            "",            # 12. 字符级 (最终 fallback)
        ]

    # ─────────────────────────────────────────────
    #  公开方法
    # ─────────────────────────────────────────────

    def split_text(self, text: str) -> list:
        """
        分割文本为多个块

        完整流程:
            text → _split() 递归分割 → _merge_with_overlap() 合并 → chunks
        """
        if not text or not text.strip():
            return []

        # 第 1 步：递归分割 → 得到许多小片段 (splits)
        splits = self._split(text.strip(), self.separators)

        # 第 2 步：合并 + 加 overlap → 得到最终块 (chunks)
        chunks = self._merge_with_overlap(splits)

        return chunks

    # ─────────────────────────────────────────────
    #  核心递归逻辑
    # ─────────────────────────────────────────────

    def _split(self, text: str, separators: list) -> list:
        """
        递归分割文本

        算法:
            1. 如果 text 已经 ≤ chunk_size → 直接返回 [text]
            2. 如果没有更多分隔符 → 按 chunk_size 硬切
            3. 用 separators[0] 切分 text
                - 如果没切出多段 → 尝试 separators[1] (降级)
                - 如果切出多段 → 对每段递归调用 _split(..., separators[1:])
        """
        # ----- 基本情况 1: 文本已经足够小 -----
        if len(text) <= self.chunk_size:
            return [text]

        # ----- 基本情况 2: 没有更多分隔符，按长度硬切 -----
        if not separators:
            return self._split_by_char(text)

        current_sep = separators[0]
        next_seps = separators[1:]

        # 用当前分隔符切分
        if current_sep:
            parts = text.split(current_sep)
        else:
            # 空分隔符 = 按字符切分
            parts = list(text)

        # 如果没切出多段 → 换更细的分隔符
        if len(parts) <= 1:
            return self._split(text, next_seps)

        # 对每段递归切分 (用更细的分隔符)
        result = []
        for part in parts:
            if not part:
                continue
            # 注意: 传入 next_seps，降级到更细的分隔符
            sub_splits = self._split(part, next_seps)
            result.extend(sub_splits)

        return result

    def _split_by_char(self, text: str) -> list:
        """
        最后的 fallback: 按字符数强制切分
        (只有在没有任何分隔符可用时才会触发)
        """
        return [
            text[i: i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]

    # ─────────────────────────────────────────────
    #  合并与重叠
    # ─────────────────────────────────────────────

    def _merge_with_overlap(self, splits: list) -> list:
        """
        将小片段合并成 chunk_size 左右的块，相邻块间加 overlap

        流程:
            current_chunk = ""
            for split in splits:
                if current_chunk + split ≤ chunk_size:
                    继续合并
                else:
                    保存 current_chunk
                    新 current_chunk = old_chunk[-overlap:] + split

        示例 (chunk_size=10, overlap=3):
            splits = ["AAAAA", "BBBBB", "CCCCC", "DDDDD"]
            块 1: "AAAAABBBBB"  (10 字符)
            块 2: "BBBCCCCCDDDDD"
                    ↑ 这是来自块 1 末尾的重叠
        """
        if not splits:
            return []

        chunks = []
        current = ""  # 正在构建的当前块

        for split in splits:
            if not current:
                # 第一个块，直接开始
                current = split
                continue

            # 如果加上当前 split 不超过 chunk_size → 合并
            if len(current) + len(split) <= self.chunk_size:
                current += split
            else:
                # 当前块已经满了 → 保存
                chunks.append(current)

                # 新块 = 旧块末尾的 overlap + 当前 split
                overlap = self._tail(current, self.chunk_overlap)
                current = overlap + split

        # 最后一块
        if current:
            chunks.append(current)

        return chunks

    @staticmethod
    def _tail(text: str, length: int) -> str:
        """取文本末尾指定长度的字符串"""
        if len(text) <= length:
            return text
        return text[-length:]


# ========== 使用示例 (可直接运行) ==========
if __name__ == "__main__":
    # ---- 测试 1: 基本分割 ----
    test_text_1 = """
## 工作流与对话流

### 1. 工作流和对话流有什么区别？
工作流（Workflow）和对话流（Bot）是 Coze 平台中两种不同的 AI 应用开发模式。
工作流适用于自动化任务处理，而对话流适用于对话式交互场景。
工作流通常处理结构化任务，对话流则处理自然语言交互。

### 2. 工作流支持哪些节点类型？
工作流支持 LLM、Code、Condition、Loop、HTTP Request 等多种节点类型。
每种节点类型都有其特定的功能和配置参数。

## 提示词优化

### 1. 如何提升提示词质量？
提供具体的上下文和示例，使用结构化格式。
明确指定输出格式可以显著提升生成质量。
"""

    print("=" * 60)
    print("✂️  递归文本分割演示")
    print("=" * 60)

    # 使用较小的参数方便观察
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    print(f"\n📐 参数配置:")
    print(f"   chunk_size    = {splitter.chunk_size}")
    print(f"   chunk_overlap = {splitter.chunk_overlap}")
    print(f"   分隔符数量   = {len(splitter.separators)} 级")

    chunks = splitter.split_text(test_text_1)
    print(f"\n📊 结果: {len(chunks)} 个块")

    for i, chunk in enumerate(chunks):
        print(f"\n{'─' * 50}")
        print(f"  块 #{i+1} (长度: {len(chunk)} 字符)")
        print(f"{'─' * 50}")
        print(chunk)

    # ---- 测试 2: 展示 overlap 效果 ----
    print("\n\n" + "=" * 60)
    print("🔄 Overlap 效果展示")
    print("=" * 60)

    # 用一段连续文字演示 overlap
    test_text_2 = "今天天气真好，适合出去散步。明天可能会下雨，记得带伞。后天多云转晴，温度适宜。"
    splitter2 = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10)
    chunks2 = splitter2.split_text(test_text_2)

    for i, chunk in enumerate(chunks2):
        print(f"\n  块 #{i+1}: 「{chunk}」")
    print(f"\n注意观察相邻块之间重复的部分，那就是 overlap 的效果——")
    print(f"确保即使切分点在句子中间，上下文也不会丢失。")

    # ---- 测试 3: 统计信息 ----
    print("\n\n" + "=" * 60)
    print("📊 块大小分布统计")
    print("=" * 60)

    splitter3 = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks3 = splitter3.split_text(test_text_1)
    sizes = [len(c) for c in chunks3]
    print(f"  块数:       {len(chunks3)}")
    print(f"  最小块:     {min(sizes)} 字符")
    print(f"  最大块:     {max(sizes)} 字符")
    print(f"  平均块大小: {sum(sizes)/len(sizes):.0f} 字符")
    print(f"  Overlap 占比: {128/512*100:.0f}%")

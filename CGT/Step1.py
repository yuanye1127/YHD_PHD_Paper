"""
步骤一
面向中文文本挖掘，针对 docx 格式的语料库完成全流程清洗、分词与结构化存储。
"""
import os
import jieba
import pickle
from docx import Document
from Tools import load_stopwords, preprocess_text, split_sentences

INPUT_DOCX = r".\语料库.docx"  # 输入语料库路径
STOPWORDS_PATH = r".\stopwords_full.txt"  # 停用词表路径。使用的汇总的停用词表
CUSTOM_DICT_PATH = r".\custom_dict.txt"  # 自定义词典路径
# 输出文件路径
OUTPUT_TXT = r".\corpus_processed.txt"
OUTPUT_PKL = r".\corpus_processed.pkl"

stopwords = load_stopwords(STOPWORDS_PATH)  # 加载停用词表

# 加载自定义词典
if CUSTOM_DICT_PATH and os.path.exists(CUSTOM_DICT_PATH):
    jieba.load_userdict(CUSTOM_DICT_PATH)
    print(f"已加载自定义词典: {CUSTOM_DICT_PATH}")
else:
    print("提示: 未找到自定义词典，使用默认分词。")

def process_docx(filepath):
    """读取 docx 文件，返回预处理后的文档列表（每个文档是一个词列表）"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"输入文件不存在: {filepath}")
    doc = Document(filepath)
    processed_docs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        sentences = split_sentences(text)  # 先切句，再逐句清洗分词
        for sent in sentences:
            words = preprocess_text(sent, stopwords)
            if words:  # 过滤掉分词结果为空的句子
                processed_docs.append(words)

    return processed_docs

# 主程序
if __name__ == '__main__':
    print("正在读取并预处理语料库...")
    docs = process_docx(INPUT_DOCX)
    print(f"共处理了 {len(docs)} 个文档（段落）。")
    # 保存为文本文件（每行一个文档，词用空格隔开）
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(' '.join(doc) + '\n')
    print(f"已保存文本格式至 {OUTPUT_TXT}")
    # 保存为 Pickle 文件（列表的列表）
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(docs, f)
    print(f"已保存 Pickle 格式至 {OUTPUT_PKL}")

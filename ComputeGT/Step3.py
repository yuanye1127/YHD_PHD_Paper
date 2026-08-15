"""
步骤三：基于LDA模型的主题推理与代表性文本提取（句子级）
包含以下功能：
1.加载步骤二训练完成的LDA主题模型及对应的词典文件。
2.读取原始语料库（.docx），按句子进行切分、清洗与分词（预处理逻辑与步骤一严格对齐）。
3.将每个句子的词袋向量输入LDA模型，推断其主导主题及对应的归属概率。
4.按主题分组，提取每个主题下概率最高的Top-N条原始短句，导出为结构化的Excel文件。
"""
import os
import jieba
import pandas as pd
from docx import Document
from gensim import corpora, models

from Tools import load_stopwords, preprocess_text, split_sentences

INPUT_DOCX = r".\语料库.docx"  # 输入语料库路径
STOPWORDS_PATH = r".\stopwords_full.txt"  # 停用词表路径。使用的汇总的中英文停用词表
CUSTOM_DICT_PATH = r".\custom_dict.txt"  # 自定义词典路径
OUTPUT_DIR = r".\lda_results"  # 输出目录
MODEL_PATH = os.path.join(OUTPUT_DIR, "lda_model.model")  # 输入模型路径
DICT_PATH = os.path.join(OUTPUT_DIR, "dictionary.dict")  # 输入字典路径
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "Topics_Text_for_LLM.xlsx")  # 输出给大模型Excel文件
TOP_N = 50  # 每个主题提取具代表性的原文条数
stopwords = load_stopwords(STOPWORDS_PATH)  # 加载停用词表

# 加载自定义词典
if CUSTOM_DICT_PATH and os.path.exists(CUSTOM_DICT_PATH):
    jieba.load_userdict(CUSTOM_DICT_PATH)
    print(f"已加载自定义词典: {CUSTOM_DICT_PATH}")
else:
    print("提示: 未找到自定义词典，使用默认分词。")

# -------- 3. 主程序 ----------
if __name__ == '__main__':
    # 加载LDA模型和词典
    lda_model = models.LdaModel.load(MODEL_PATH)
    dictionary = corpora.Dictionary.load(DICT_PATH)
    # 读取原始语料,并按句子切分
    doc = Document(INPUT_DOCX)
    raw_sentences = []  # 存储切分后的短句原文
    processed_docs = []  # 存储对应的分词列表
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # 将长段落切分为短句列表
        sentences = split_sentences(text)
        for sent in sentences:
            words = preprocess_text(sent, stopwords)  # 对短句进行分词
            if words:  # 只有分词结果非空才保留
                raw_sentences.append(sent)  # 保存短句原文
                processed_docs.append(words)  # 保存短句的分词
    print(f"成功切分并预处理，共生成 {len(raw_sentences)} 条短句样本。")
    corpus_bow = [dictionary.doc2bow(doc) for doc in processed_docs]  # 将预处理后的文档转为BOW向量
    # 计算每个短句的主导主题与对应归属概率，同时过滤空BOW无效样本
    topic_assignments = []
    topic_probs = []
    valid_raw = []
    valid_processed = []
    for sent, words, bow in zip(raw_sentences, processed_docs, corpus_bow):
        # 边界兜底：空BOW直接跳过，避免max()报错
        if not bow:
            continue
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        dominant_topic, max_prob = max(topics, key=lambda x: x[1])
        topic_assignments.append(dominant_topic)
        topic_probs.append(max_prob)
        valid_raw.append(sent)
        valid_processed.append(words)
    print(f"过滤空样本后，有效可分类样本共 {len(valid_raw)} 条。")
    # 构建DataFrame
    df = pd.DataFrame({
        '原始文本': valid_raw,
        '分词列表': [' '.join(doc) for doc in valid_processed],
        '主导主题': topic_assignments,
        '主题概率': topic_probs
    })
    print(f"\n各主题短句分布:\n{df['主导主题'].value_counts().sort_index()}")
    # 提取每个主题重要短句原文
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        for topic_id in range(lda_model.num_topics):
            subset = df[df['主导主题'] == topic_id].head(TOP_N)
            if subset.empty:
                continue
            subset[['原始文本']].to_excel(writer, sheet_name=f'主题{topic_id}', index=False)
            print(f"主题 {topic_id} 已提取 {len(subset)} 条短句。")


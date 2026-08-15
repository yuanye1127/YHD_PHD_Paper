import os
import re
import jieba
import pickle
import pandas as pd


def load_stopwords(filepath):
    # 加载停用词表
    if not os.path.exists(filepath):
        print(f"警告: 停用词文件未找到 '{filepath}'，将使用默认停用词。")
        return set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上',
                    '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
    with open(filepath, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])


# 预处理函数
def preprocess_text(text, stopwords):
    """清洗并分词，返回词列表（加强版：剔除单字、英文、链接）"""
    # 清洗：因为是纯中文政策文本。只保留中文、数字，去掉所有英文字母。
    text = re.sub(r'[a-zA-Z]', ' ', text)  # 去掉所有英文字母
    text = re.sub(r'[^\u4e00-\u9fa5\d]', ' ', text)  # 只保留中文和数字
    words = jieba.cut(text, cut_all=False)      # 分词：使用jieba库进行分词
    # 二次清洗：长度必须大于1，不能全是数字，不能包含链接关键词，且不在停用词表
    cleaned = []
    for w in words:
        w = w.strip()
        if len(w) <= 1:
            continue
        if w.isdigit():
            continue
        if w in stopwords:
            continue
        cleaned.append(w)
    return cleaned


def split_sentences(text):
    # 按中文标点切分句子
    # 按 。 ！ ？ ； 以及换行符切分，过滤掉太短的碎片
    parts = re.split(r'[。！？；\n]+', text)
    # 保留长度大于10个字符的片段，并去除首尾空格
    return [p.strip() for p in parts if len(p.strip()) > 10]


def load_corpus(pkl_path):
    """加载预处理后的语料库"""
    with open(pkl_path, 'rb') as f:
        docs = pickle.load(f)
    print(f"成功加载语料库，共 {len(docs)} 个文档。")
    return docs


def print_and_save_topics(lda_model, num_topics, output_csv):
    """
    打印各主题关键词并保存为 CSV（包含权重）
    每个主题生成两列：主题N_词 和 主题N_权重
    """
    print("\n=== 各主题 Top 10 关键词及权重 ===")
    data = {}
    for idx in range(num_topics):
        # 获取主题词及其权重（元组列表）
        words_probs = lda_model.show_topic(idx, topn=30)
        # 分别提取词和权重
        words = [w for w, _ in words_probs]
        probs = [p for _, p in words_probs]
        # 控制台打印（带权重，保留4位小数）
        print(f"主题 {idx}: " + ", ".join([f"{w}({p:.4f})" for w, p in words_probs]))
        # 存入字典，用于构建DataFrame
        data[f'主题{idx}_词'] = words
        data[f'主题{idx}_权重'] = probs

    # 构建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n主题关键词及权重已保存至: {output_csv}")


def build_fewshot_prompt(examples, prompt_info):
    prompt = prompt_info
    for i, ex in enumerate(examples, 1):
        prompt += f"\n示例{i}：\n输入：{ex['原始文本']}\n输出：{ex['人工编码']}\n"
    prompt += "\n现在，请严格按照以上格式，对以下文本进行编码，只输出代码：\n"
    return prompt
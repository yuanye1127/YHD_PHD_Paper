"""
步骤二
语料预处理后的第二步分析，依赖 step1 输出的分词后语料
包含以下功能：
1.最优主题数选择：持遍历指定主题数范围，计算 C_V 主题一致性得分并绘制变化曲线，确定最优主题数量
2.模型训练：可配置词典极端词过滤阈值、训练迭代次数与随机种子，训练指定主题数的最终LDA主题模型
3.结果保存：导出主题-关键词结果，保存LDA模型与词典文件，支持后续复用与二次分析
4.交互可视化：生成可视化pyLDAvis交互式HTML页面，直观呈现主题间距离、词项分布与主题占比
"""
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os

from Tools import load_corpus, print_and_save_topics

# 输入.pkl文件路径，由step1.py生成
INPUT_PKL = r".\corpus_processed.pkl"
# 输出目录
OUTPUT_DIR = r".\lda_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 输出文件名
COHERENCE_PLOT = os.path.join(OUTPUT_DIR, "coherence_scores.png")
LDA_VIS_HTML = os.path.join(OUTPUT_DIR, "lda_visualization.html")
TOPIC_WORDS_CSV = os.path.join(OUTPUT_DIR, "topic_words.csv")

# 模型参数
TOPIC_RANGE = range(2, 11)   # 主题数量范围
BEST_TOPIC_NUM = 6  # 最终使用的主题数，根据曲线图调整
SKIP_COHERENCE = False  # 是否跳过一致性得分计算。确认最终使用主题数后候使用True，训练最终模型。
# 训练迭代参数
PASSES = 20  # 对语料库遍历学习的次数。
ITERATIONS = 60  # 每次遍历中参数收敛的最大迭代次数。
RANDOM_STATE = 42  # 随机种子

# 词典清洗参数
NO_BELOW = 4  # 过滤掉在少于该数值的文档中出现的词
NO_ABOVE = 0.9  # 过滤掉出现在超过该比例文档中的词

def compute_coherence_scores(processed_docs, topic_range, passes, iterations, random_state, no_below, no_above):
    """计算不同主题数下的一致性得分，并返回词典、语料和模型列表"""
    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    coherence_values = []
    model_list = []
    for num_topics in topic_range:
        print(f"正在训练{num_topics}个主题的模型...")
        model = models.LdaModel(corpus=corpus,id2word=dictionary,num_topics=num_topics,
                                random_state=random_state,passes=passes,iterations=iterations)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model,texts=processed_docs,dictionary=dictionary,coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print(f"主题数{num_topics}: 一致性得分={coherence_values[-1]:.4f}")
    return dictionary, corpus, model_list, coherence_values

def train_final_model(processed_docs, num_topics, passes, iterations, random_state, no_below, no_above):
    """使用指定的主题数训练最终 LDA 模型"""
    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = models.LdaModel(corpus=corpus,id2word=dictionary,num_topics=num_topics,
                                random_state=random_state,passes=passes,iterations=iterations,per_word_topics=True)
    return dictionary, corpus, lda_model

if __name__ == '__main__':
    processed_docs = load_corpus(INPUT_PKL)  # 加载语料
    if not SKIP_COHERENCE:  # 判断是否跳过一致性得分计算
        print("\n开始计算不同主题数下的一致性得分...")
        dictionary, corpus, model_list, coherence_values = (
            compute_coherence_scores(processed_docs, TOPIC_RANGE, PASSES, ITERATIONS, RANDOM_STATE,NO_BELOW, NO_ABOVE))
        # 绘制一致性曲线
        plt.figure(figsize=(10, 6))
        plt.plot(TOPIC_RANGE, coherence_values, marker='o')
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.title("Topic Consistency Score Change Chart")
        plt.grid(True)
        plt.savefig(COHERENCE_PLOT, dpi=900, bbox_inches='tight')
        plt.show()
        print(f"一致性得分曲线已保存至:{COHERENCE_PLOT}")
        best_k = TOPIC_RANGE[coherence_values.index(max(coherence_values))]  # 找出最佳主题数
        print(f"\n最佳主题数（按一致性得分）: {best_k}，得分 = {max(coherence_values):.4f}")
    else:
        print(f"\n跳过一致性得分计算，直接使用主题数 {BEST_TOPIC_NUM} 训练最终模型。")

    final_k = BEST_TOPIC_NUM  # 跳过一致性得分计算，直接计算最终模型
    print(f"\n正在使用主题数 {final_k} 训练最终LDA模型...")
    final_dict, final_corpus, final_model = (
        train_final_model(processed_docs, final_k, PASSES, ITERATIONS, RANDOM_STATE, NO_BELOW, NO_ABOVE))

    print_and_save_topics(final_model, final_k, TOPIC_WORDS_CSV)  # 输出并保存主题词

    model_path = os.path.join(OUTPUT_DIR, "lda_model.model")  # 保存训练好的模型
    final_model.save(model_path)
    print(f"LDA模型已保存至: {model_path}")

    dict_path = os.path.join(OUTPUT_DIR, "dictionary.dict")  # 保存词典
    final_dict.save(dict_path)
    print(f"词典已保存至: {dict_path}")

    vis_data = gensimvis.prepare(final_model, final_corpus, final_dict)  # 生成pyLDAvis可视化
    pyLDAvis.save_html(vis_data, LDA_VIS_HTML)
    print(f"pyLDAvis 可视化已保存至: {LDA_VIS_HTML}")



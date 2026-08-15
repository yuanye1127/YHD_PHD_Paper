"""
步骤四
基于大语言模型的主题编码/开放式编码与一致性评估
"""
import re
import time
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from Tools import build_fewshot_prompt

# 输入文件：包含“原始文本”、“人工编码”两列（以及“是否作为示例”列）
INPUT_EXCEL = r".\Manual_Coding_Base.xlsx"
# 输出文件：脚本运行后生成的检验结果，包含相似度得分。
OUTPUT_EXCEL = r".\Validation_Results.xlsx"
# 语义编码器模型路径
ENCODER_MODEL = r".\e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
# Ollama 大模型调用参数
MODEL_NAME = "deepseek-r1:7b"  # 模型名称
TEMPERATURE = 0.2  # 温度参数：控制输出的随机性。
TOP_P = 0.9  # top_p 采样阈值：保留累积概率达到 p 的最小词集，默认 0.9 是常用值。
RANDOM_SEED = 42  # 随机种子 (seed)：
MAX_TOKENS = 128  # 最大生成 token 数
API_TIMEOUT = 60  # API 超时时间（秒）
REQUEST_INTERVAL = 0.3  # 两次请求之间的间隔（秒）
# 示例与检验参数
MAX_EXAMPLES = 5  # Few-shot 示例的最大数量
USE_EXAMPLE_COLUMN = True  # 是否使用“是否作为示例”列来手动指定示例
# 相似度判断阈值
THRESHOLD_HIGH = 0.85
THRESHOLD_MID = 0.70
OLLAMA_URL = "http://localhost:11434/api/generate"  # 拼接 Ollama API 地址
# 提示词模板
SYSTEM_PROMPT = """你是一名精通扎根理论的城市风险沟通研究编码助手。
任务：对给定的原始文本进行开放式编码，生成简短的、名词性的代码（4-8个字）。
规则：
1. 聚焦"风险沟通效果"（影响因素、机制、结果、对策）。
2. 代码必须简洁，不要写成句子。
3. 只输出代码本身，不要输出任何解释、序号或标点。
以下是几个示例（输入 -> 输出）：
"""

def call_llm(prompt_text):
    """调用本地 Ollama API，返回生成的代码"""
    payload = {
        "model": MODEL_NAME,"prompt": prompt_text,"stream": False,"temperature": TEMPERATURE,
        "seed": RANDOM_SEED,"max_tokens": MAX_TOKENS,"top_p": TOP_P
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=API_TIMEOUT)
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    line = re.sub(r'^输出[：:]|^\d+\.\s*', '', line)
                    return line.strip()
            return lines[0].strip() if lines else "解析失败"
        else:
            return f"HTTP错误: {response.status_code}"
    except Exception as e:
        return f"请求异常: {str(e)}"

def compute_similarity(manual_codes, llm_codes):
    """计算两组编码列表的余弦相似度"""
    if not manual_codes or not llm_codes:
        return 0.0, []
    if not hasattr(compute_similarity, "encoder"):
        print("正在加载语义编码器模型（本地缓存，无需联网）...")
        compute_similarity.encoder = SentenceTransformer(ENCODER_MODEL, local_files_only=True)
    encoder = compute_similarity.encoder
    manual_clean = [str(c) if pd.notna(c) else "无编码" for c in manual_codes]
    llm_clean = [str(c) if pd.notna(c) else "无编码" for c in llm_codes]
    manual_emb = encoder.encode(manual_clean, convert_to_numpy=True)
    llm_emb = encoder.encode(llm_clean, convert_to_numpy=True)
    similarities = []
    for i in range(len(manual_clean)):
        sim = cosine_similarity([manual_emb[i]], [llm_emb[i]])[0][0]
        similarities.append(float(sim))
    avg_sim = np.mean(similarities) if similarities else 0.0
    return avg_sim, similarities

#  主程序
if __name__ == "__main__":
    df = pd.read_excel(INPUT_EXCEL)  # 读取数据
    required_cols = ['原始文本', '人工编码']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Excel 中缺少必需列: {col}")
    # 划分 Few-shot 示例和测试集
    if USE_EXAMPLE_COLUMN and '是否作为示例' in df.columns:
        examples_df = df[df['是否作为示例'] == '是']
        test_df = df[df['是否作为示例'] != '是'].copy()
    else:
        examples_df = df.head(3)
        test_df = df.iloc[3:].copy()
    if len(examples_df) > MAX_EXAMPLES:
        examples_df = examples_df.head(MAX_EXAMPLES)
    examples = examples_df[['原始文本', '人工编码']].to_dict('records')
    print(f"\n使用 {len(examples)} 条 Few-shot 示例。")
    print(f"待检验文本数量: {len(test_df)} 条。")
    # 构建提示词
    fewshot_prompt = build_fewshot_prompt(examples, SYSTEM_PROMPT)
    # 批量调用 LLM
    llm_codes = []
    for idx, row in test_df.iterrows():
        text = row['原始文本']
        if pd.isna(text) or not str(text).strip():
            llm_codes.append("文本为空")
            continue
        full_prompt = fewshot_prompt + f"\n输入：{text}\n输出："
        result_code = call_llm(full_prompt)
        llm_codes.append(result_code)
        print(f"  进度: {idx+1}/{len(test_df)}  ->  {result_code}")
        time.sleep(REQUEST_INTERVAL)
    test_df['LLM编码'] = llm_codes
    # 计算相似度
    print("\n正在计算语义相似度...")
    avg_sim, sim_list = compute_similarity(
        test_df['人工编码'].tolist(),
        test_df['LLM编码'].tolist()
    )
    test_df['相似度得分'] = sim_list
    # 输出统计结果
    print("检验结果汇总!!!")
    print(f"检验样本数: {len(test_df)}")
    print(f"平均余弦相似度: {avg_sim:.4f}")
    print(f"最高相似度: {max(sim_list):.4f}")
    print(f"最低相似度: {min(sim_list):.4f}")
    print(f"标准差: {np.std(sim_list):.4f}")
    if avg_sim >= THRESHOLD_HIGH:
        print("\n✅ 结论: 一致性极高 (>= 0.85) —— 编码方案非常稳定，具有良好的可重复性。")
    elif avg_sim >= THRESHOLD_MID:
        print("\n⚠️ 结论: 一致性中等 (0.70-0.85) —— 编码方案基本稳定，建议检查差异较大的样本。")
    else:
        print("\n❌ 结论: 一致性较低 (< 0.70) —— 建议重新审视编码规则或增加 Few-shot 示例。")
    # 保存结果
    examples_df['LLM编码'] = '（示例，未调用模型）'
    examples_df['相似度得分'] = np.nan
    final_df = pd.concat([examples_df, test_df], ignore_index=True)
    cols = ['原始文本', '人工编码', 'LLM编码', '相似度得分']
    if '是否作为示例' in final_df.columns:
        cols = ['是否作为示例'] + cols
    final_df = final_df[cols]
    final_df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n详细结果已保存至: {OUTPUT_EXCEL}")
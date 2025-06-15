from openai import OpenAI
import json
import time

# ====================== 配置区 ====================== #
client = OpenAI(
    api_key="Your siliconflow api key",
    base_url="https://api.siliconflow.cn/v1"
)

MODEL = "Your fine-tuned model ID on siliconflow"

SYSTEM_PROMPT = (
    "你现在是一个细粒度片段级仇恨言论识别系统。请严格按照下列格式抽取四元组："
    "“评论对象 | 论点 | 目标群体 | 是否仇恨 [END]”。\n"
    "如有多个四元组，用[SEP]分隔。所有元素之间严格用\" | \"分割，结尾用[END]。分隔符和空格不能省略。\n"
    "- 评论对象（Target）：帖子的评述对象，如一个人或一个群体。当实例无具体目标时设为NULL。\n"
    "- 论点（Argument）：包含对评论目标关键论点的信息片段。\n"
    "- 目标群体（Targeted Group）：只能为以下5类其中之一：“地域（Region）”、“种族（Racism）”、“性别（Sexism）”、“LGBTQ（LGBTQ）”、“其他（others）”。\n"
    "- 是否仇恨（Hateful）：如包含仇恨言论则为hate，否则为non-hate。\n"
    "【注意事项】\n"
    "1. 不要省略任何元素或分隔符。\n"
    "2. 多四元组请严格用[SEP]分隔。\n"
    "3. 顺序必须保持“评论对象 | 论点 | 目标群体 | 是否仇恨 [END]”。"
)
#按需替换为你自己的系统提示词，注意与训练时一致
# =================================================== #


def infer_one(text: str, model: str = MODEL, max_retry: int = 3) -> str:
    """单条推理，返回去除换行的字符串；重试 max_retry 次"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text}
    ]
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,     # 抽取任务通常不需要 4096
                temperature=0.2      # 略低温度，保证格式稳定
            )
            content = resp.choices[0].message.content
            return content.replace("\n", " ").replace("\r", " ").strip()
        except Exception as e:
            print(f"[{attempt+1}/{max_retry}] Error: {e}")
            if attempt < max_retry - 1:
                time.sleep(2)
            else:
                return ""


if __name__ == "__main__":
    with open("test1.json", encoding="utf-8") as fr:
        data = json.load(fr)

    results = []
    for idx, sample in enumerate(data, 1):
        text = sample["content"]
        result = infer_one(text)
        results.append(result)

        print(f"第{idx}条 完成")
        print("输入：", text)
        print("输出：", result)
        print("=" * 30)

    with open("demo.txt", "w", encoding="utf-8") as fw:
        for line in results:
            fw.write(line + "\n")

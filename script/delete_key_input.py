import os
import json
import sys

def remove_trailing_spaces_and_newlines(input_str):
    # 从后往前遍历字符串
    for i in range(len(input_str) - 1, -1, -1):
        if input_str[i] not in [' ', '\n']:
            # 找到第一个非"/n"和空格的字符，然后返回该位置之前的部分
            return input_str[:i + 1]

    # 如果字符串全是"/n"和空格，则返回空字符串
    return ""
def read_jsonl_files(directory):
    """
    读取指定目录下的所有.jsonl文件，并将它们合并到一个列表中。
    """
    data = []
    for file in os.listdir(directory):
        if file.endswith(".jsonl"):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                for line in f:
            
                    data.append(json.loads(line))
    return data

def filter_data_n(data):
    for i in range(len(data)):
        data[i]["instruction"]=remove_trailing_spaces_and_newlines(data[i]["instruction"])
    return data
def filter_data_key(data, keywords):
    """
    遍历列表，检查每个元素的'instruction'关键字对应的value是否包含指定的关键词。
    如果包含，则从列表中删除该元素。
    """
    return [item for item in data if not any(keyword in item.get('instruction', '') for keyword in keywords)]

def split_and_save_data(data, directory, num_files=128):
    """
    将列表平均分成指定数量的小列表，并将这些小列表保存为.jsonl文件。
    """
    chunk_size = len(data) // num_files
    for i in range(num_files):
        chunk = data[i*chunk_size:(i+1)*chunk_size]
        with open(os.path.join(directory, f'{i}.jsonl'), 'w', encoding='utf-8') as f:
            for item in chunk:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

# 示例使用
source_dir = sys.argv[1]
target_dir = sys.argv[2]
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
keywords = ["新闻", "材料","图片","链接","网址","根据","简历","简介","文章","摘要","文本","提取","关键词","网站"]  # 指定字符列表

# 读取文件
data = read_jsonl_files(source_dir)
# 根据关键词过滤数据
data = filter_data_key(data, keywords)
# 删除句末空白和换行符
data = filter_data_n(data)
split_and_save_data(data, target_dir)

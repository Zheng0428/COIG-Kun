import os
import json
def detect_repetition_improved(text, threshold=5):
    """
    Detects if there is a repetition of a substring in the given text.
    This function is improved to handle cases where the repeated phrases are not separated by spaces.
    If the same substring repeats more than 'threshold' times consecutively, 
    returns True and the repeated substring.
    """
    # Finding the smallest repeating unit
    for i in range(1, len(text)):
        unit = text[:i]
        if text.startswith(unit * (threshold + 1)):
            return True, unit
    return False, None

def detect_repetition_improved_v1(text, threshold=5):
    """
    Detects if there is a repetition of a substring in the given text.
    This function is improved to handle cases where the repeated phrases are not separated by spaces
    and can be composed of multiple characters.
    If the same substring repeats more than 'threshold' times consecutively, 
    returns True and the repeated substring.
    """
    # Iterate over the length of potential repeating units
    for i in range(1, len(text) // threshold + 1):
        # Iterate through the text to check for repeating units
        for j in range(len(text) - i * threshold + 1):
            unit = text[j:j + i]
            count = 0
            k = j
            # Count the number of times the unit repeats consecutively
            while k + i <= len(text) and text[k:k + i] == unit:
                count += 1
                k += i
            
            # Check if repetition exceeds the threshold
            if count > threshold:
                return True, unit
    return False, None

def process_text(text):
    """
    Processes the text by checking for continuous spaces or newlines (combined) longer than four characters.
    If found, the function returns the part of the sentence before the long space or newline.

    Args:
    text (str): The input text to be processed.

    Returns:
    str: Processed text with the decision and the processed sentence.
    """
    # Search for continuous spaces or newlines (combined) longer than four characters
    match = re.search(r'[\s\n]{5,}', text)
    if match:
        # Find the position where the long space or newline starts
        position = match.start()
        # Return the part of the sentence before the long space or newline
        return True, text[:position].strip()
    else:
        # If no long space or newline is found, return the original text
        return False, text.strip()

def remove_duplicates(text):
    """
    Removes duplicate sentences from the input text and returns a flag indicating
    if duplicates were found, the text with duplicates removed, and a dictionary
    of duplicate sentences with their counts.
    """
    # Splitting the text into sentences
    sentences = text.split("\n")

    # Removing any extra spaces and filtering out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # Counting duplicates using a dictionary
    sentence_counts = {}
    for sentence in sentences:
        if sentence in sentence_counts:
            sentence_counts[sentence] += 1
        else:
            sentence_counts[sentence] = 1

    # Check if duplicates were found
    duplicates_found = any(count > 1 for count in sentence_counts.values())
    if duplicates_found:
        duplicate_sentences = [sentence for sentence, count in sentence_counts.items() if count > 1]

        final_output = max(duplicate_sentences, key=len)
        # duplicate_sentences = {sentence: count for sentence, count in sentence_counts.items() if count > 1}

        return duplicates_found, final_output
    else:
        return duplicates_found, text

def process_jsonl_files(all_data, output_file):
    # 确保输出文件夹存在
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    for line in all_data:
        data = json.loads(line)
        del data["input"]
        if len(data['output'])<20:  #output长度小于20的删除
            continue
        found,_ =detect_repetition_improved(data['output'])
        if found: #回答中出现灾难重复，删除
            a = 1
            continue
        instruction = data.get("instruction", "")
        error_found, instruction = process_text(instruction)
        if error_found:
            # json.dump(data, outfile, ensure_ascii=False)
            # outfile.write('\n')
            # json.dump("1 "+instruction, outfile, ensure_ascii=False)
            # outfile.write('\n')
            data["instruction"] = instruction
            json.dump(data, outfile, ensure_ascii=False)
        else:
            duplicates_found, fix_instruction = remove_duplicates(instruction)
            if duplicates_found:
                # json.dump(data, outfile, ensure_ascii=False)
                # outfile.write('\n')
                if fix_instruction == instruction[:len(fix_instruction)]:
                    # json.dump('2 '+fix_instruction, outfile, ensure_ascii=False)
                    # outfile.write('\n')
                    data["instruction"] = fix_instruction
                    json.dump(data, outfile, ensure_ascii=False)
                else:  #不要了
                    continue
                    json.dump('3 '+fix_instruction, outfile, ensure_ascii=False)
                    outfile.write('\n')
            else:
                json.dump(data, outfile, ensure_ascii=False)
        
        outfile.write('\n')

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
        data[i]["instruction"]=remove_trailing_spaces_and_newlines(data[i]["output"])
    return data
def filter_data_key(data, keywords):
    """
    遍历列表，检查每个元素的'instruction'关键字对应的value是否包含指定的关键词。
    如果包含，则从列表中删除该元素。
    """
    return [item for item in data if not any(keyword in item.get('output', '') for keyword in keywords)]

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
source_dir = sys.argv[1]
target_dir = sys.argv[2]
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

keywords = ["作为", "根据","知识库","2023","抱歉"]  # 指定字符列表

# 读取文件
data = read_jsonl_files(source_dir)
# 根据关键词过滤数据
data = filter_data_key(data, keywords)
# 删除句末空白和换行符
data = filter_data_n(data)
process_jsonl_files(data, target_dir)

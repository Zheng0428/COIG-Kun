import re
from collections import Counter
import os,json


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

def process_jsonl_files(folder_path, output_file):
    # 确保输出文件夹存在
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    with open(output_file, 'w') as outfile:
        # 遍历文件夹中的所有jsonl文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    for line in file:
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


# folder_path = '/ML-A100/home/tianyu/Kun/data/final_kun/wudao'
# output_file = '/ML-A100/home/tianyu/Kun/data/final_kun/wudao.jsonl'
folder_path = '/ML-A100/home/tianyu/Kun/data/skypile_kun/vllm_filter_Yi_34B'
output_file = '/ML-A100/home/tianyu/Kun/data/final_kun/skypile.jsonl'
process_jsonl_files(folder_path, output_file)
print ('done')
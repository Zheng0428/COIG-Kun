import json
import os
import sys

def read_jsonl_files(directory):
    """Reads all .jsonl files in the specified directory and returns a list of dictionaries."""
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line.strip())
                    all_data.append(data)
    return all_data

def filter_data(data_list):
    """Filters out dictionaries that do not contain '【好】' in their keys."""
    all_data=[]
    for data in data_list:
        if "【好】" in data["score"] and "【差】" not in data["score"]:
            temp={
                "instruction":data["instruction"],
                "output":data["output"]
            }
            all_data.append(temp)
    return all_data

def split_and_save(data_list, target_directory, num_files=128):
    """Splits the data list into `num_files` parts and saves them as .jsonl files in the target directory."""
    split_size = len(data_list) // num_files
    for i in range(num_files):
        start_index = i * split_size
        end_index = None if i == num_files - 1 else start_index + split_size
        filename = f"{i}.jsonl"
        with open(os.path.join(target_directory, filename), 'w', encoding='utf-8') as file:
            for data in data_list[start_index:end_index]:
                json.dump(data, file,ensure_ascii=False)
                file.write('\n')

# Example usage
source_directory = sys.argv[1]  # replace with the path to the source directory
target_directory = sys.argv[2]  # replace with the path to the target directory
if not os.path.exists(target_directory):
    os.makedirs(target_directory)
# Read, filter, split, and save data
data = read_jsonl_files(source_directory)
filtered_data = filter_data(data)
split_and_save(filtered_data, target_directory)

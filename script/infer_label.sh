folder_path="/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/Humpback-CH/data/WuDaoCorpus2.0_base_200G/"
model="baichuan-chat"
# 遍历文件夹中的所有文件
for file in "${folder_path}"/*
do
  # 获取文件名（不包含路径）
  filename="$(basename "${file}")"

  # 使用文件名作为参数运行Python脚本
  python gen_model_answer.py\
  --model-path "/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/Humpback-CH/checkpoint/base_13b"\
  --model-id  "$model" \
  --data-path folder_path \
  --output-path "/cpfs/29cd2992fe666f2a/user/huangwenhao/xw/Humpback-CH/data/Kun/LabeledData/" \
  --data-id "${filename}"
done
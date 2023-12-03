LABEL_MODEL_PATH=$1
POINT_MODEL_PATH=$2
ANSWER_MODEL_PATH=$3
DATA_PATH=$4
OUTPUT_PATH=$5

sh infer_label.sh LABEL_MODEL_PATH DATA_PATH label_data

python delete_key_input.py label_data delete_key_data

sh infer_point.sh POINT_MODEL_PATH delete_key_data point_data

python select_data.py point_data select_data_input

sh infer_label.sh ANSWER_MODEL_PATH select_data_input answer_data

python delete_key_output.py answer_data select_data_output

python deplicate.py select_data_output OUTPUT_PATH

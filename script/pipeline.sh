CARD_ID=$1
LABEL_MODEL_PATH=$2
POINT_MODEL_PATH=$3
ANSWER_MODEL_PATH=$4
DATA_PATH=$5
OUTPUT_PATH=$6

sh infer_label.sh CARD_ID LABEL_MODEL_PATH DATA_PATH label_data

python delete_key_input.py label_data delete_key_data

sh infer_point.sh CARD_ID POINT_MODEL_PATH delete_key_data point_data

python select_data.py point_data select_data_input

sh infer_label.sh CARD_ID ANSWER_MODEL_PATH select_data_input answer_data

python delete_key_output.py answer_data select_data_output

python deplicate.py select_data_output OUTPUT_PATH

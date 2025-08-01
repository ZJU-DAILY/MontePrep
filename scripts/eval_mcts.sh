
python src/utils/evaluator.py \
    --json_folder result/auto_pipeline/qwen_32B/execution\
    --data_folder data/auto_pipeline \
    --output_base predict/auto_pipeline/qwen_32B/execution \
    --length_types 1 \
    --start_num 0 \
    --end_num 101

# Complete experiment:
# for auto_pipeline --length_type 1 2 3 4 5 6 9 \
# for buidlings  --length_type 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
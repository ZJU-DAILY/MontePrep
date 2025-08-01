
python src/main.py \
    --base_path data/auto_pipeline \
    --result_dir result/auto_pipeline/qwen_32B/execution \
    --length_type 1 \
    --start_num 0 \
    --end_num 101 \
    --log_path logs/mcts_qwen_32B_execution.txt



# Complete experiment:
# for auto_pipeline --length_type 1 2 3 4 5 6 9 \
# for buidlings  --length_type 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
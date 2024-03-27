# ARG
python main.py \
    --seed 3759 \
    --gpu 0 \
    --lr 2e-5 \
    --model_name ARG \
    --language ch \
    --root_path /path/to/zh-data \
    --bert_path /path/to/chinese-bert-wwm-ext \
    --data_name zh-arg \
    --data_type rationale \
    --rationale_usefulness_evaluator_weight 2.2 \
    --llm_judgment_predictor_weight 1.8 

# # ARG-D
# python main.py \
#     --seed 3759 \
#     --gpu 0 \
#     --lr 5e-4 \
#     --model_name ARG-D \
#     --language ch \
#     --root_path /path/to/zh-data \
#     --bert_path /path/to/chinese-bert-wwm-ext \
#     --data_name zh-argd \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 2.2 \
#     --llm_judgment_predictor_weight 1.8 \
#     --kd_loss_weight 15 \
#     --teacher_path ./param_model/ARG_zh-arg/1/parameter_bert.pkl
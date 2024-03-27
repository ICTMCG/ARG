# ARG
python main.py \
    --seed 3759 \
    --gpu 1 \
    --lr 5e-5 \
    --model_name ARG \
    --language en \
    --root_path /path/to/en-data \
    --bert_path /path/to/bert-base-uncased \
    --data_name en-arg \
    --data_type rationale \
    --rationale_usefulness_evaluator_weight 1.5 \
    --llm_judgment_predictor_weight 1.0

# # ARG-D
# python main.py \
#     --seed 3759 \
#     --gpu 1 \
#     --lr 5e-5 \
#     --model_name ARG-D \
#     --language en \
#     --root_path /path/to/en-data \
#     --bert_path /path/to/bert-base-uncased \
#     --data_name en-argd \
#     --data_type rationale \
#     --rationale_usefulness_evaluator_weight 1.5 \
#     --llm_judgment_predictor_weight 1.0 \
#     --kd_loss_weight 1.0 \
#     --teacher_path ./param_model/ARG_en-arg/1/parameter_bert.pkl
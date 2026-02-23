#!/bin/bash

# Activate the conda environment
echo "🔄 Activating conda environment: blink371"
conda activate blink371

# Set variables
OUTPUT_PATH="original/path/Latin_NEL/LT4HALA/$1/"
TRANSFORMER_PATH="original/path/test_for_NEL/xlm-roberta-base"
VERSION=$3

echo "📁 Output path set to: $OUTPUT_PATH"
echo "🧠 Transformer path: $TRANSFORMER_PATH"
echo "📦 Version: $VERSION"

# Determine data path and text length type
if [ "$2" = "kurz" ]; then
    ORIGINAL_DATA_PATH="blink_data/$VERSION/final_split/Kurz"
    IS_KURZ=True
else
    ORIGINAL_DATA_PATH="blink_data/$VERSION/final_split/Voll"
    IS_KURZ=False

fi

echo "📂 Original data path: $ORIGINAL_DATA_PATH"
echo "✂️ Text length type: $2 (IS_KURZ=$IS_KURZ)"

# #Train the biencoder
# echo "🚀 Starting biencoder training..."
# python blink_path/BLINK/blink/biencoder/train_biencoder.py \
#     --data_path "${ORIGINAL_DATA_PATH}" \
#     --output_path "${OUTPUT_PATH}biencoder" \
#     --bert_model "$TRANSFORMER_PATH" \
#     --mode train \
#     --max_context_length 128 \
#     --max_cand_length 128 \
#     --num_train_epochs 5 \
#     --learning_rate 2e-6 \
#     --print_interval 50 \
#     --eval_interval 2000 \
#     --train_batch_size 32 \
#     --eval_batch_size 32 \
#     --seed 42 \
#     --path_to_model "original/path/biencoder/pytorch_model.bin" \


# echo "✅ Biencoder training complete."

# Evaluate the biencoder
# echo "🧪 Evaluating biencoder..."
# python blink_path/BLINK/blink/biencoder/eval_biencoder.py \
#     --data_path "$ORIGINAL_DATA_PATH" \
#     --path_to_model "${OUTPUT_PATH}biencoder/pytorch_model.bin" \
#     --bert_model "$TRANSFORMER_PATH" \
#     --max_context_length 128 \
#     --max_cand_length 128 \
#     --top_k 64 \
#     --output_path "${OUTPUT_PATH}biencoder_results" \
#     --mode train,valid \
#     --save_topk_result \
#     --cand_encode_path "${OUTPUT_PATH}cand_encoding" \
#     --cand_pool_path "${OUTPUT_PATH}cand_pool" \
#     --re_entities "v5,"$2

# echo "✅ Biencoder evaluation complete."

# # Train the crossencoder
echo "🚀 Starting crossencoder training..."
python blink_path/BLINK/blink/crossencoder/train_cross.py \
    --data_path "${OUTPUT_PATH}/biencoder_results/top64_candidates" \
    --output_path "${OUTPUT_PATH}crossencoder" \
    --bert_model "$TRANSFORMER_PATH" \
    --mode train \
    --max_context_length 128 \
    --max_cand_length 128 \
    --max_seq_length 256 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --top_k 64 \
    --add_linear \
    --eval_batch_size 1 \
    --train_batch_size 1 \
    --eval_interval 1000 \
    --save_interval 1000 \
    --print_interval 100 \
    --path_to_model original/path/crossencoder/epoch_1/pytorch_model.bin \


# echo "✅ Crossencoder training complete."
# echo "🎉 All steps finished successfully!"

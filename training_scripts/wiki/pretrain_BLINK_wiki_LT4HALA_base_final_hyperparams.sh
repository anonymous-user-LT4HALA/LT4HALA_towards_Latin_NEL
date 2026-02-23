#!/bin/bash

# Activate the conda environment
echo "🔄 Activating conda environment: blink371"
conda activate blink371

# Set variables
OUTPUT_PATH="/output_path/$1/"
TRANSFORMER_PATH="xlm-roberta-base"

echo "📁 Output path set to: $OUTPUT_PATH"
echo "🧠 Transformer path: $TRANSFORMER_PATH"
echo "📦 Version: $VERSION"
# Determine data path and text length type
if [ "$2" = "kurz" ]; then
    ORIGINAL_DATA_PATH="/home/nobackup/wikipedia/la/v4/Kurz"
    IS_KURZ=True
else
    ORIGINAL_DATA_PATH="/home/nobackup/wikipedia/la/v4/Voll"
    IS_KURZ=False
fi

echo "📂 Original data path: $ORIGINAL_DATA_PATH"
echo "✂️ Text length type: $2 (IS_KURZ=$IS_KURZ)"

Train the biencoder
echo "🚀 Starting biencoder training..."
python BLINK/blink/biencoder/train_biencoder.py \
    --data_path "${ORIGINAL_DATA_PATH}_batches_32" \
    --shuffle True \
    --output_path "${OUTPUT_PATH}biencoder" \
    --bert_model "$TRANSFORMER_PATH" \
    --mode train \
    --max_context_length 128 \
    --max_cand_length 128 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --print_interval 50 \
    --eval_interval 200 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --seed 22 \


echo "✅ Biencoder training complete."

#Evaluate the biencoder
echo "🧪 Evaluating biencoder..."
python BLINK/blink/biencoder/eval_biencoder.py \
    --data_path "$ORIGINAL_DATA_PATH" \
    --path_to_model "${OUTPUT_PATH}biencoder/pytorch_model.bin" \
    --bert_model "$TRANSFORMER_PATH" \
    --max_context_length 128 \
    --max_cand_length 128 \
    --top_k 64 \
    --output_path "${OUTPUT_PATH}biencoder_results" \
    --mode train,valid \
    --save_topk_result \
    --cand_encode_path "${OUTPUT_PATH}cand_encoding" \
    --cand_pool_path "${OUTPUT_PATH}cand_pool" \
    --re_entities "wiki_$2" \

# echo "✅ Biencoder evaluation complete."

# Train the crossencoder
echo "🚀 Starting crossencoder training..."
python BLINK/blink/crossencoder/train_cross.py \
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

# echo "✅ Crossencoder training complete."
# echo "🎉 All steps finished successfully!"

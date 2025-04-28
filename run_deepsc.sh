#!/bin/bash

# run_deepsc.sh - Script for training and evaluating DeepSC with and without SemRect
# Usage: ./run_deepsc.sh [train|test|all] [standard|semrect]

# Default parameters
VOCAB_FILE="europarl/vocab.json"
CHANNEL="AWGN"  # AWGN, Rayleigh, or Rician
D_MODEL=128
DFF=512
NUM_LAYERS=4
NUM_HEADS=8
BATCH_SIZE=128
EPOCHS=50
SNR=10
EPSILON=0.1
SEMRECT_EPOCHS=30

# Directories
STANDARD_CHECKPOINT="checkpoints/deepsc-standard"
SEMRECT_CHECKPOINT="checkpoints/deepsc-semrect"

# Function to show usage
show_usage() {
    echo "Usage: ./run_deepsc.sh [train|test|all] [standard|semrect]"
    echo ""
    echo "Commands:"
    echo "  train    : Train the model"
    echo "  test     : Test the model performance"
    echo "  attack   : Test against adversarial attacks"
    echo "  all      : Train and test"
    echo ""
    echo "Model Type:"
    echo "  standard : Standard DeepSC"
    echo "  semrect  : DeepSC with SemRect integration"
    echo ""
    echo "Examples:"
    echo "  ./run_deepsc.sh train standard   # Train standard DeepSC"
    echo "  ./run_deepsc.sh train semrect    # Train DeepSC with SemRect"
    echo "  ./run_deepsc.sh test standard    # Test standard DeepSC"
    echo "  ./run_deepsc.sh attack           # Test both models against adversarial attacks"
    echo "  ./run_deepsc.sh all semrect      # Train and test DeepSC with SemRect"
}

# Function to train standard DeepSC
train_standard() {
    echo "Training standard DeepSC model..."
    python main.py \
        --vocab-file "$VOCAB_FILE" \
        --checkpoint-path "$STANDARD_CHECKPOINT" \
        --channel "$CHANNEL" \
        --d-model $D_MODEL \
        --dff $DFF \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS
}

# Function to train DeepSC with SemRect
train_semrect() {
    echo "Training DeepSC with SemRect integration..."
    python main.py \
        --vocab-file "$VOCAB_FILE" \
        --checkpoint-path "$SEMRECT_CHECKPOINT" \
        --channel "$CHANNEL" \
        --d-model $D_MODEL \
        --dff $DFF \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --use-semrect \
        --semrect-epochs $SEMRECT_EPOCHS
}

# Function to test standard DeepSC performance
test_standard() {
    echo "Testing standard DeepSC performance..."
    python performance.py \
        --vocab-file "$VOCAB_FILE" \
        --checkpoint-path "$STANDARD_CHECKPOINT" \
        --channel "$CHANNEL" \
        --d-model $D_MODEL \
        --dff $DFF \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS
}

# Function to test DeepSC with SemRect performance
test_semrect() {
    echo "Testing DeepSC with SemRect performance..."
    python performance.py \
        --vocab-file "$VOCAB_FILE" \
        --checkpoint-path "$SEMRECT_CHECKPOINT" \
        --channel "$CHANNEL" \
        --d-model $D_MODEL \
        --dff $DFF \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS
}

# Function to test against adversarial attacks
test_attacks() {
    echo "Testing both models against adversarial attacks..."
    python performance.py \
        --vocab-file "$VOCAB_FILE" \
        --checkpoint-path "$STANDARD_CHECKPOINT" \
        --channel "$CHANNEL" \
        --d-model $D_MODEL \
        --dff $DFF \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS \
        --test-attacks \
        --epsilon $EPSILON \
        --semrect-checkpoint "$SEMRECT_CHECKPOINT/final_with_semrect.pth"
}

# Check for command line arguments
if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

COMMAND=$1
MODEL_TYPE=$2

# Process command
case $COMMAND in
    train)
        case $MODEL_TYPE in
            standard)
                train_standard
                ;;
            semrect)
                train_semrect
                ;;
            *)
                echo "Error: Invalid model type. Use 'standard' or 'semrect'."
                show_usage
                exit 1
                ;;
        esac
        ;;
    test)
        case $MODEL_TYPE in
            standard)
                test_standard
                ;;
            semrect)
                test_semrect
                ;;
            *)
                echo "Error: Invalid model type. Use 'standard' or 'semrect'."
                show_usage
                exit 1
                ;;
        esac
        ;;
    attack)
        test_attacks
        ;;
    all)
        case $MODEL_TYPE in
            standard)
                train_standard
                test_standard
                ;;
            semrect)
                train_semrect
                test_semrect
                ;;
            *)
                echo "Error: Invalid model type. Use 'standard' or 'semrect'."
                show_usage
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Error: Invalid command. Use 'train', 'test', 'attack', or 'all'."
        show_usage
        exit 1
        ;;
esac

echo "Done!" 
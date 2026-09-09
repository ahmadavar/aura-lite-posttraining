#!/bin/bash
# This script waits for GRPO training to finish, then runs evaluation.
# No internet needed. Runs entirely on your laptop.
# Usage: bash scripts/finish_pipeline.sh

cd /Users/ahmadavar/Desktop/aura-lite-posttraining

TRAIN_PID=48621

echo "=== PIPELINE RUNNER ==="
echo "Waiting for GRPO training (PID $TRAIN_PID) to finish..."

# Wait for training to complete
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
done

echo ""
echo "Training finished at $(date)"
echo ""

# Check if model was saved
if [ ! -f models/trained_rl/adapter_model.safetensors ]; then
    echo "ERROR: No trained model found at models/trained_rl/"
    echo "Training may have crashed. Check results/grpo_full_training_log.txt"
    exit 1
fi

echo "Model found. Starting evaluation..."
echo ""

# Evaluate RL model on test set
echo "=== Evaluating RL on TEST set ==="
PYTHONPATH=. python3.11 scripts/evaluate.py --condition rl --split test 2>&1 | tee results/rl_test_eval_log.txt

echo ""
echo "=== Evaluating RL on CHALLENGE set ==="
PYTHONPATH=. python3.11 scripts/evaluate.py --condition rl --split challenge 2>&1 | tee results/rl_challenge_eval_log.txt

echo ""
echo "=== ALL DONE ==="
echo "Finished at $(date)"
echo ""
echo "Results saved to:"
echo "  results/rl_test_results.json"
echo "  results/rl_challenge_results.json"
echo ""
echo "To see results: cat results/rl_test_results.json | python3.11 -m json.tool"

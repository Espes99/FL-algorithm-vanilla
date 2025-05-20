cp learning_params.py learning_params.py.bak

rounds=(2 5 10 15 20 25)
clients=(2 5 10 15 20 25)

for round in "${rounds[@]}"; do
  for client in "${clients[@]}"; do
    echo "Running CKKS simulation with $round rounds and $client clients"

    cat > learning_params.py << EOF

NUM_ROUNDS = $round
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_CLIENTS = $client

METHODS = ["PLAIN", "CKKS", "ABHO"]
EOF

    python ckks_fl_pipeline.py

    # If the run was unsuccessful, report and continue, 0 success
    if [ $? -ne 0 ]; then
    echo "Error running ckks_fl_pipeline.py with $round rounds and $client clients"
    fi
  done
done

# Restore the original learning_params.py
mv learning_params.py.bak learning_params.py

echo "All runs completed!"
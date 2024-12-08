#!/bin/bash

# first need to start server:
# python3 start_server.py > output_030324_02.log 2>&1 &

REPS=1
# Start measuring time
start_time=$(date +%s.%N)

# # Run your Python script
# python3 testing/main.py -i onnx -r $REPS | tee -a results/out_onnx.log;
# # Calculate the elapsed time
# end_time=$(date +%s.%N)
# elapsed_time=$(echo "$end_time - $start_time" | bc)
# # Display the elapsed time
# echo "Time taken: $elapsed_time seconds"

# python3 testing/main.py -i ov -r $REPS | tee -a results/out_ov.log;
# # Calculate the elapsed time
# end_time=$(date +%s.%N)
# elapsed_time=$(echo "$end_time - $start_time" | bc)
# # Display the elapsed time
# echo "Time taken: $elapsed_time seconds"

python3 testing/main.py -i torchscript -r $REPS | tee -a results/out_torchscript.log;
# Calculate the elapsed time
end_time=$(date +%s.%N)
elapsed_time=$(echo "$end_time - $start_time" | bc)
# Display the elapsed time
echo "Time taken: $elapsed_time seconds"

# python3 testing/main.py -i torch -r $REPS | tee -a results/out_torch.log;
# # Calculate the elapsed time
# end_time=$(date +%s.%N)
# elapsed_time=$(echo "$end_time - $start_time" | bc)
# # Display the elapsed time
# echo "Time taken: $elapsed_time seconds"
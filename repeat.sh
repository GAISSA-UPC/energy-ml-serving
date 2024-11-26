#!/bin/bash

# nohup ./repeat.sh > repeat.out 2>&1 & 

# Define the number of repetitions
N=10 #[CHANGE]

pkill -f runall_update.sh
pkill -f $python3
pkill -f '/home/fjdur/EnergiBridge/target/release/energibridge'

rm -r results
mkdir results

# Loop to execute the script n times
for ((i=1; i<=$N; i++)); do
    echo "Running repetition $i"
    echo "$(date "+%Y-%m-%d %H:%M:%S")"
    
    # Start the script in the background and redirect output to a file
    nohup ./runall_update.sh > results/runall_$i.out 2>&1 &
    PID=$!  # Get the process ID of the background process
    
    # Wait for the process to finish
    wait $PID
    
    # Move 'results' directory to 'results_i', where i is the repetition number
    mv results results_$i
    mkdir results

done

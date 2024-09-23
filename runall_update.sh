#!/bin/bash

# nohup python3 start_server.py >  results/nohup.out 2>&1 &
# nohup ./runall.sh > results/runall.out 2>&1 &
# nohup ./runall.sh ov > results/runall.out 2>&1 &
# windows: ./runall.sh > results/runall.out 2>&1
# nohup ./runall_update.sh > results/runall.out 2>&1 &

# set true if need to install from scratch
INSTALL=false
# set true to run experiments with all runtime engines
ALL_RUNTIME_ENGINES=true
# set true if server needs to be started from this script
START_SERVER=true

SECONDS=0
WAIT_BETWEEN_RUNTIME=300 # [CHANGE] Wait time between running experiments with each runtime engine
REPS=1 # Number of repetitions for each runtime engine experiment
IDLE_TIME=300 # [CHANGE] 300 to measure idle resource consumption

day=$(date +%d)
month=$(date +%m)
year=$(date +%Y)
SERVER_LOG="results/server_$day$month$year_01_$1.log"
#python3=/home/usuaris/fduran/Python-3.8.4/python # rdlab local python
python3=python3 # comment if rdlab, gaissa
#python3=python # personal setup
TOKENIZERS_PARALLELISM=true
OMP_NUM_THREADS=8

#energibridge="/d/GAISSA/EnergiBridge-main/target/release/energibridge.exe" # windows
energibridge="/home/fjdur/EnergiBridge/target/release/energibridge" # linux

runtime_engines=('onnx' 'torch' 'torchscript') # if  CUDAExecutionProvider # [CHANGE]
#runtime_engines=('torch' 'onnx' 'torchscript' 'ov' ) #if CPUExecutionProvider  # [CHANGE]

#runtime_engines=('torchscript' ) #'ov'
#models=('codeparrot-small' 'pythia-410m') # 'codet5-base' 'codet5p-220' 
models=('pythia1-4b' 'tinyllama' 'codeparrot-small' 'pythia-410m' 'phi2') #  [CHANGE] 'phi2'
#models=('phi2') # [CHANGE]


mkdir results
# Print all processes with ps aux
echo "All processes beginning:" >> results/processes.log
ps aux >> results/processes.log

# Function to echo with a prefix
print() {
    local prefix="[$0] "
    echo "$prefix $1"
}



if $INSTALL = true; then
    print "Installing..."
    # Update package information
    sudo apt-get update

    # Upgrade installed packages
    sudo apt-get upgrade -y
    #sudo apt-get -y install uvicorn
    #pip3 install unicorn --user 

    $python3 -m pip install --upgrade pip
    $python3 -m pip install -r requirements.txt
    wget https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
fi

print "Initializating server... $python3"

if [ $START_SERVER = true ]; then
    # check uvicorn is not running: sudo lsof -t -i tcp:8000 | xargs kill -9
    print "Initializating server..."
    lsof -t -i tcp:8000 | xargs kill -9
    #uvicorn app.api_code:app --host 0.0.0.0 --port 8000 > output.log 2>&1 &
    
    #uvicorn app.api_code:app  --host 0.0.0.0 --port 8000   > output_01.log 2>&1 & #  --reload --reload-dir app
    $python3 start_server.py >  $SERVER_LOG 2>&1 &
    
fi

#nvidia-smi -i 1 --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -l 1 -f output_gpu
# Start GPU monitoring with nvidia-smi and log to a temporary file
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory --format=csv -l 1 > gpu_log.csv &
#nvidia_smi_command = f"nvidia-smi -i {GPU_ID} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -l {GPU_SMI_SEC} -f {GPU_RESULTS}"

#nvidia_smi_command = f"nvidia-smi -i {GPU_ID} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms {GPU_SMI_MS} -f {gpu_metrics}"

# nvidia command
#nvidia-smi -i 0 --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms 100 -f "output_gpu.csv" &

# Save the PID of the nvidia-smi process
#NVIDIA_PID=$!

# python testing/main.py -i onnx -r 1 -m 'codet5-base' | tee -a results/out_$runtime.log;
#timestamps=()

./kill_nvidia.sh

# Initialize a file for timestamps
timestamps_file="results/runall_timestamps.csv"
echo "timestamp,runtime,model" > "$timestamps_file"

load_times_file="results/load_times.csv"
echo "start_time,end_time,engine,model_name" > "$load_times_file"


if [ $ALL_RUNTIME_ENGINES = true ]; then
    
    start_date=$(date "+%Y-%m-%d %H:%M:%S")  # Capture start date in a consistent format for CSV
    echo "$start_date,-,IDLE" >> "$timestamps_file"  # Record start of runtime
    print "sleeping $IDLE_TIME seconds"
    date
    my_command="sleep $IDLE_TIME"
    $energibridge --output "results/energy_idle.csv"  --interval 200 $my_command
    date

    # Main loop over runtime engines
    for runtime in "${runtime_engines[@]}"; do

        start_time=$(date +%s.%N)
        print "---------------------------------------------------------------"
        print "| Running experiments, RUNTIME ENGINE -> $runtime |"
        print "---------------------------------------------------------------"
        
        start_date=$(date "+%Y-%m-%d %H:%M:%S")  # Capture start date in a consistent format for CSV
        echo "$start_date,$runtime,START" >> "$timestamps_file"  # Record start of runtime

        #timestamps+=("Start: $(date)")
        #timestamps+=($(date))


        # Inner loop over models
        for model in "${models[@]}"; do
            print "Testing with $runtime and $model"

            # nvidia command
            nvidia_output="results/nvidia_${model}_${runtime}.csv"
            nvidia-smi -i 0 --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms 100 -f $nvidia_output &

            #Save the PID of the nvidia-smi process
            NVIDIA_PID=$!

            my_command="python3 testing/main.py -i $runtime -r $REPS -m $model"
            $energibridge --output "results/energy_${model}_${runtime}.csv" --command-output results/out_${model}_${runtime}.log --interval 200 $my_command
            
            kill $NVIDIA_PID
            
            #timestamps+=($(date))
            model_date=$(date "+%Y-%m-%d %H:%M:%S")  # Capture timestamp after model run
            echo "$model_date,$runtime,$model" >> "$timestamps_file"  # Record model run
            sync; echo 1 > /proc/sys/vm/drop_caches

        done

        # Calculate the elapsed time
        end_time=$(date +%s.%N)
        elapsed_time=$(echo "$end_time - $start_time" | bc)
        #print "Timestamps: ${timestamps[@]}" # num_runtimes *( num_models +1)
        print "Time taken for $runtime: $elapsed_time seconds"

        # Wait between runs if required
        print "WAIT_BETWEEN_RUNTIME..."
        while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
            sleep 1
        done
    done
fi

finish_date=$(date "+%Y-%m-%d %H:%M:%S")  # Capture start date in a consistent format for CSV
echo "$finish_date,-,FINISH" >> "$timestamps_file"  # Record start of runtime

# Optionally wait for the Python script to finish
#wait $!

# Kill the nvidia-smi monitoring process
#kill $NVIDIA_PID


print "---------------------------------------------------------------"
print "runall.sh finished!!!"
print "Finishing server..."
lsof -t -i tcp:8000 | xargs kill -9
print "---------------------------------------------------------------"
#print "Timestamps: ${timestamps[@]}"
#echo "Timestamps: ${timestamps[@]}"

print "Settings:"
print "ALL_RUNTIME_ENGINES=$ALL_RUNTIME_ENGINES"
print "RUNTIME_ENGINE=$1"
print "REPS=$REPS"
print "WAIT_BETWEEN_RUNTIME=$WAIT_BETWEEN_RUNTIME"
print "IDLE_TIME=$IDLE_TIME"
print "results dir information: (wc -l results/*):"
wc -l results/*

# Print all processes with ps aux
echo "All processes end:" >> results/processes.log
ps aux >> results/processes.log
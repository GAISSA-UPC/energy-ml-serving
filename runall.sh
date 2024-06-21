#!/bin/bash

# nohup python3 start_server.py >  results/nohup.out 2>&1 &
# nohup ./runall.sh > results/runall.out 2>&1 &
# nohup ./runall.sh ov > results/runall.out 2>&1 &
# windows: ./runall.sh > results/runall.out 2>&1

# set true if need to install from scratch
INSTALL=false
# set true to run experiments with all runtime engines
ALL_RUNTIME_ENGINES=true
# set true if server needs to be started from this script
START_SERVER=true

SECONDS=0
WAIT_BETWEEN_RUNTIME=5 # Wait time between running experiments with each runtime engine
REPS=1 # Number of repetitions for each runtime engine experiment
IDLE_TIME=10 #300 to measure idle resource consumption

day=$(date +%d)
month=$(date +%m)
year=$(date +%Y)
SERVER_LOG="results/server_$day$month$year_01_$1.log" #change
#python3=/home/usuaris/fduran/Python-3.8.4/python # rdlab local python
python3=python3 # comment if rdlab, gaissa
#python3=python # personal setup
TOKENIZERS_PARALLELISM=true
OMP_NUM_THREADS=8

#energibridge="/d/GAISSA/EnergiBridge-main/target/release/energibridge.exe" # windows
energibridge="/home/fjdur/EnergiBridge/target/release/energibridge" # linux

runtime_engines=('torch' 'onnx' 'ov' 'torchscript')
models=('codeparrot-small' 'pythia-410m') # 'codet5-base' 'codet5p-220' 

# Function to echo with a prefix
print() {
    local prefix="[$0] "
    echo "$prefix $1"
}

# Verify your python version
# PYTHON3_VER=$($python3 --version 2>&1 | awk '{print $2}')
# PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
# print "python3 version: "
# print "$PYTHON3_VER"
# print "python version: "
# print "$PYTHON_VER"
# Verify your python version
# PYTHON3_VER=$($python3 --version 2>&1 | awk '{print $2}')
# PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
# print "python3 version: "
# print "$PYTHON3_VER"
# print "python version: "
# print "$PYTHON_VER"

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
# nvidia command
#nvidia-smi -i 0 --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms 100 -f "output_gpu.csv" &

# Save the PID of the nvidia-smi process
#NVIDIA_PID=$!

# python testing/main.py -i onnx -r 1 -m 'codet5-base' | tee -a results/out_$runtime.log;
timestamps=()

if [ $ALL_RUNTIME_ENGINES = true ]; then
    print "sleeping $IDLE_TIME seconds"
    date
    sleep $IDLE_TIME
    date
    print "USING ALL RUNTIME ENGINES"    
    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "TORCH" |"
    print "---------------------------------------------------------------"
    start_time=$(date +%s.%N)
    date
    
    timestamps+=($(date))
    runtime=${runtime_engines[0]}
    model=${models[0]}
    #$python3 testing/main.py -i $runtime -r $REPS -m $model | tee -a results/out_$runtime.log;
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command
    tail -10 results/gpu_results.csv 
    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codet5p-220' | tee -a results/out_$runtime.log;
    model=${models[1]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command
    tail -10 results/gpu_results.csv 
    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codeparrot-small' | tee -a results/out_$runtime.log;
    model=${models[2]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command
    tail -10 results/gpu_results.csv 
    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'pythia-410m' | tee -a results/out_$runtime.log;
    model=${models[3]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command
    tail -10 results/gpu_results.csv 
    timestamps+=($(date))

    # Calculate the elapsed time
    end_time=$(date +%s.%N)
    #elapsed_time=$(echo "$end_time - $start_time" | bc)
    elapsed_time=$(print "$end_time - $start_time" )
    print "Timestamps: ${timestamps[@]}"
    # Display the elapsed time
    print "Time taken for $runtime: $elapsed_time seconds"
    
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "ONNX" |"
    print "---------------------------------------------------------------"
    start_time=$(date +%s.%N)
    date
    #$python3 testing/main.py -i onnx -r $REPS | tee -a results/out_onnx.log;
    timestamps+=($(date))
    runtime=${runtime_engines[1]}
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codet5-base' | tee -a results/out_$runtime.log;
    model=${models[0]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command
    tail -10 results/gpu_results.csv 
    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codet5p-220' | tee -a results/out_$runtime.log;
    model=${models[1]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command
    tail -10 results/gpu_results.csv 
    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codeparrot-small' | tee -a results/out_$runtime.log;
    model=${models[2]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command
    tail -10 results/gpu_results.csv 
    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'pythia-410m' | tee -a results/out_$runtime.log;
    model=${models[3]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command
    tail -10 results/gpu_results.csv 
    timestamps+=($(date))
    # Calculate the elapsed time
    end_time=$(date +%s.%N)
    #elapsed_time=$(echo "$end_time - $start_time" | bc)
    elapsed_time=$(print "$end_time - $start_time" )
    echo "Timestamps: ${timestamps[@]}"

    # Display the elapsed time
    echo "Time taken for $runtime: $elapsed_time seconds"

    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "OV" |"
    print "---------------------------------------------------------------"
    start_time=$(date +%s.%N)
    date
    #$python3 testing/main.py -i ov -r $REPS | tee -a results/out_ov.log;
    timestamps+=($(date))
    runtime=${runtime_engines[2]}
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codet5-base' | tee -a results/out_$runtime.log;
    model=${models[0]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command

    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codet5p-220' | tee -a results/out_$runtime.log;
    model=${models[1]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command

    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codeparrot-small' | tee -a results/out_$runtime.log;
    model=${models[2]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command

    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'pythia-410m' | tee -a results/out_$runtime.log;
    model=${models[3]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command

    timestamps+=($(date))
    # Calculate the elapsed time
    end_time=$(date +%s.%N)
    #elapsed_time=$(echo "$end_time - $start_time" | bc)
    elapsed_time=$(echo "$end_time - $start_time" )
    echo "Timestamps: ${timestamps[@]}"
    # Display the elapsed time
    echo "Time taken for $runtime: $elapsed_time seconds"

    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "TORCHSCRIPT" |"
    print "---------------------------------------------------------------"
    start_time=$(date +%s.%N)
    date
    date
    #$python3 testing/main.py -i torchscript -r $REPS | tee -a results/out_torchscript.log;
    timestamps+=($(date))
    runtime=${runtime_engines[3]}
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codet5-base' | tee -a results/out_$runtime.log;
    model=${models[0]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command

    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codet5p-220' | tee -a results/out_$runtime.log;
    model=${models[1]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command

    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'codeparrot-small' | tee -a results/out_$runtime.log;
    model=${models[2]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command

    timestamps+=($(date))
    #$python3 testing/main.py -i $runtime -r $REPS -m 'pythia-410m' | tee -a results/out_$runtime.log;
    model=${models[3]}
    my_command="$python3 testing/main.py -i $runtime -r $REPS -m $model"
    $energibridge --output results/energy_$runtime-$model.csv --command-output results/out_$runtime-$model.log --interval 200 $my_command

    timestamps+=($(date))

    # # Calculate the elapsed time
    end_time=$(date +%s.%N)
    #elapsed_time=$(echo "$end_time - $start_time" | bc)
    elapsed_time=$(echo "$end_time - $start_time" )
    echo "Timestamps: ${timestamps[@]}"
    # Display the elapsed time
    echo "Time taken for $runtime: $elapsed_time seconds"

else
    print "USING JUST $1 AS RUNTIME ENGINE"
    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "$1" |"
    print "---------------------------------------------------------------"
    start_time=$(date +%s.%N)
    
    $python3 testing/main.py -i $1 -r $REPS -m 'codeparrot-small' | tee -a results/out_$1.log;
    $python3 testing/main.py -i $1 -r $REPS -m 'pythia-410m' | tee -a results/out_$1.log;
    $python3 testing/main.py -i $1 -r $REPS -m 'codet5p-220' | tee -a results/out_$1.log;
    $python3 testing/main.py -i $1 -r $REPS -m 'codet5-base' | tee -a results/out_$1.log;


    # Calculate the elapsed time
    end_time=$(date +%s.%N)
    # elapsed_time=$(echo "$end_time - $start_time" | bc)
    elapsed_time=$(echo "$end_time - $start_time" )
    # Display the elapsed time
    print "Time taken: $elapsed_time seconds"
    print "Time taken: $elapsed_time seconds"
fi


# Optionally wait for the Python script to finish
wait $!

# Kill the nvidia-smi monitoring process
#kill $NVIDIA_PID


print "---------------------------------------------------------------"
print "runall.sh finished!!!"
print "Finishing server..."
lsof -t -i tcp:8000 | xargs kill -9
print "---------------------------------------------------------------"
print "Timestamps: ${timestamps[@]}"
echo "Timestamps: ${timestamps[@]}"

print "Settings:"
print "ALL_RUNTIME_ENGINES=$ALL_RUNTIME_ENGINES"
print "RUNTIME_ENGINE=$1"
print "REPS=$REPS"
print "WAIT_BETWEEN_RUNTIME=$WAIT_BETWEEN_RUNTIME"
print "IDLE_TIME=$IDLE_TIME"
print "results dir information: (wc -l results/*):"
wc -l results/*

    

#!/bin/bash

# nohup python3 start_server.py >  results/nohup.out 2>&1 &
# nohup ./runall.sh > results/runall.out 2>&1 &
# nohup ./runall.sh ov > results/runall.out 2>&1 &
# nohup ./runall.sh ov > results/runall.out 2>&1 &

# set true if need to install from scratch
INSTALL=false
# set true to run experiments with all runtime engines
ALL_RUNTIME_ENGINES=true
# set true if server needs to be started from this script
START_SERVER=true

SECONDS=0
WAIT_BETWEEN_RUNTIME=10 # Wait time between running experiments with each runtime engine
REPS=1 # Number of repetitions for each runtime engine experiment

day=$(date +%d)
month=$(date +%m)
year=$(date +%Y)

SERVER_LOG="results/output_$day$month$year_01_$1.log" #change

python3=/home/usuaris/fduran/Python-3.8.4/python
python3=python3 # comment if rdlab

# Function to echo with a prefix
print() {
    local prefix="[$0] "
    echo "$prefix $1"
}

PYTHON3_VER=$($python3 --version 2>&1 | awk '{print $2}')
PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')

print "python3 version: "
print "$PYTHON3_VER"
print "python version: "
print "$PYTHON_VER"

if $INSTALL = true; then
    print "Installing..."
    # Update package information
    sudo apt-get update

    # Upgrade installed packages
    sudo apt-get upgrade -y
    sudo apt-get -y install uvicorn
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

if [ $ALL_RUNTIME_ENGINES = true ]; then
    print "USING ALL RUNTIME ENGINES"
    
    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "TORCH" |"
    print "---------------------------------------------------------------"
    start_time=$(date +%s.%N)
    #$python3 testing/main.py -i torch -r $REPS | tee -a results/out_torch.log;
    runtime="torch"
    $python3 testing/main.py -i $runtime -r $REPS -m 'codet5-base' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'codet5p-220' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'codeparrot-small' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'pythia-410m' | tee -a results/out_$runtime.log;

    # Calculate the elapsed time
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    # Display the elapsed time
    echo "Time taken: $elapsed_time seconds"
    
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "ONNX" |"
    print "---------------------------------------------------------------"
    start_time=$(date +%s.%N)
    #$python3 testing/main.py -i onnx -r $REPS | tee -a results/out_onnx.log;
    runtime="onnx"
    $python3 testing/main.py -i $runtime -r $REPS -m 'codet5-base' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'codet5p-220' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'codeparrot-small' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'pythia-410m' | tee -a results/out_$runtime.log;
    # Calculate the elapsed time
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    # Display the elapsed time
    echo "Time taken: $elapsed_time seconds"

    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "OV" |"
    print "---------------------------------------------------------------"
    start_time=$(date +%s.%N)
    #$python3 testing/main.py -i ov -r $REPS | tee -a results/out_ov.log;
    runtime="ov"
    $python3 testing/main.py -i $runtime -r $REPS -m 'codet5-base' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'codet5p-220' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'codeparrot-small' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'pythia-410m' | tee -a results/out_$runtime.log;
    # Calculate the elapsed time
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    # Display the elapsed time
    echo "Time taken: $elapsed_time seconds"

    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "TORCHSCRIPT" |"
    print "---------------------------------------------------------------"
    start_time=$(date +%s.%N)
    #$python3 testing/main.py -i torchscript -r $REPS | tee -a results/out_torchscript.log;
    runtime="torchscript"
    $python3 testing/main.py -i $runtime -r $REPS -m 'codet5-base' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'codet5p-220' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'codeparrot-small' | tee -a results/out_$runtime.log;
    $python3 testing/main.py -i $runtime -r $REPS -m 'pythia-410m' | tee -a results/out_$runtime.log;

    # Calculate the elapsed time
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    # Display the elapsed time
    echo "Time taken: $elapsed_time seconds"

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
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    # Display the elapsed time
    echo "Time taken: $elapsed_time seconds"
fi

print "---------------------------------------------------------------"
print "runall.sh finished!!!"
print "Finishing server..."
lsof -t -i tcp:8000 | xargs kill -9
print "---------------------------------------------------------------"
print "Settings:"
print "ALL_RUNTIME_ENGINES=$ALL_RUNTIME_ENGINES"
print "RUNTIME_ENGINE=$1"
print "REPS=$REPS"
print "results dir information: (wc -l results/*):"
wc -l results/*

    

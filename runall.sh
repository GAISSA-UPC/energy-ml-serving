#!/bin/bash

# set true if need to install from scratch
INSTALL=true
# set true to run experiments with all runtime engines
ALL_RUNTIME_ENGINES=false
SECONDS=0
WAIT_TIME=120  # Wait time to server to be up
WAIT_SERVER_IS_UP=5 # A time after server is up
WAIT_BETWEEN_RUNTIME=20 # Wait time between running experiments with each runtime engine
REPS=1 # Number of repetitions for each runtime engine experiment

START_SERVER=true # set true if server needs to be started from this script
SERVER_HOST="localhost"
SERVER_PORT="8000"
CHECK_ENDPOINT="/"  # Adjust to your actual health check endpoint

# Function to echo with a prefix
print() {
    local prefix="[$0] "
    echo "$prefix $1"
}

python3=/home/usuaris/fduran/Python-3.8.4/python

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
fi


if [ $START_SERVER = true ]; then
    # check uvicorn is not running: sudo lsof -t -i tcp:8000 | xargs kill -9
    print "Initializating server..."
    lsof -t -i tcp:8000 | xargs kill -9
    #uvicorn app.api_code:app --host 0.0.0.0 --port 8000 > output.log 2>&1 &
    
    #uvicorn app.api_code:app  --host 0.0.0.0 --port 8000   > output_01.log 2>&1 & #  --reload --reload-dir app
    $python3 start_server.py > output_220224_01.log 2>&1 &
fi

#nvidia-smi -i 1 --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -l 1 -f output_gpu

# Function to check if the server is up
check_server() {
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://${SERVER_HOST}:${SERVER_PORT}${CHECK_ENDPOINT})
    if [ "${HTTP_STATUS}" = "200" ]; then
        print "Server is UP"
        return 0
    else
        print "Server is not yet UP (HTTP Status: ${HTTP_STATUS})"
        return 1
    fi
}


print "Checking if the server is up..."
while [ "${SECONDS}" -lt "${WAIT_TIME}" ]; do
    print "Checking if the server is up..."
    date
    #check_server && break
    sleep 50
done

#if [ "${SECONDS}" -ge "${WAIT_TIME}" ]; then
#    print "Timeout: Server did not start within ${WAIT_TIME} seconds."
#    exit 1
#fi

print "WAIT_SERVER_UP..."
while [ "${SECONDS}" -lt "${WAIT_SERVER_IS_UP}" ]; do
    date
    print "wait_server_up"
    sleep 50
done
print "Server is up. Proceeding with further steps..."


reps=$REPS

if [ $ALL_RUNTIME_ENGINES = true ]; then
    print "USING ALL RUNTIME ENGINES"
    
    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "TORCH" |"
    print "---------------------------------------------------------------"
    python3 testing/main.py -i torch -r $reps | tee -a results/out_torch.log;
    
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "ONNX" |"
    print "---------------------------------------------------------------"
    #python3 testing/main.py -i onnx -r $reps | tee -a results/out_onnx.log;
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "OV" |"
    print "---------------------------------------------------------------"
    #python3 testing/main.py -i ov -r $reps | tee -a results/out_ov.log;
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "TORCHSCRIPT" |"
    print "---------------------------------------------------------------"
    #python3 testing/main.py -i torchscript -r $reps | tee -a results/out_torchscript.log;
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

else
    print "USING JUST $1 AS RUNTIME ENGINE"
    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "$1" |"
    print "---------------------------------------------------------------"
    $python3 testing/main.py -i onnx -m 'codet5-base' -r $reps | tee -a results/out_onnx.log;
fi


    

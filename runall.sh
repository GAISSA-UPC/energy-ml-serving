#!/bin/bash

# set true if need to install from scratch
INSTALL=false
# set true to run experiments with all runtime engines
ALL_RUNTIME_ENGINES=false
SECONDS=0
WAIT_TIME=120  # Wait time to server to be up
WAIT_SERVER_IS_UP=5 # A time after server is up
WAIT_BETWEEN_RUNTIME=20 # Wait time between running experiments with each runtime engine
REPS=3 # Number of repetitions for each runtime engine experiment

START_SERVER=false # set true if server needs to be started from this script
SERVER_HOST="localhost"
SERVER_PORT="8000"
CHECK_ENDPOINT="/"  # Adjust to your actual health check endpoint

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
    sudo apt-get -y install uvicorn

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi


if [ $START_SERVER = true ]; then
    # check uvicorn is not running: sudo lsof -t -i tcp:8000 | xargs kill -9
    print "Initializating server..."
    #uvicorn app.api_code:app --host 0.0.0.0 --port 8000 > output.log 2>&1 &
    uvicorn app.api_code:app  --host 0.0.0.0 --port 8000  --reload --reload-dir app > output.log 2>&1 &
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
    #check_server && break
    sleep 1
done

if [ "${SECONDS}" -ge "${WAIT_TIME}" ]; then
    print "Timeout: Server did not start within ${WAIT_TIME} seconds."
    exit 1
fi

print "WAIT_SERVER_UP..."
while [ "${SECONDS}" -lt "${WAIT_SERVER_IS_UP}" ]; do
    sleep 1
done
print "Server is up. Proceeding with further steps..."


reps=$REPS

if [ $ALL_RUNTIME_ENGINES = true ]; then
    print "USING ALL RUNTIME ENGINES"
    
    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "TORCH" |"
    print "---------------------------------------------------------------"
    python testing/main.py -i torch -r $reps | tee -a results/out_torch.log;
    
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "ONNX" |"
    print "---------------------------------------------------------------"
    python testing/main.py -i onnx -r $reps | tee -a results/out_onnx.log;
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "OV" |"
    print "---------------------------------------------------------------"
    python testing/main.py -i ov -r $reps | tee -a results/out_ov.log;
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "TORCHSCRIPT" |"
    print "---------------------------------------------------------------"
    python testing/main.py -i torchscript -r $reps | tee -a results/out_torchscript.log;
    print "WAIT_BETWEEN_RUNTIME..."
    while [ "${SECONDS}" -lt "${WAIT_BETWEEN_RUNTIME}" ]; do
        sleep 1
    done

else
    print "USING JUST $1 AS RUNTIME ENGINE"
    print "---------------------------------------------------------------"
    print "| Running experiments, RUNTIME ENGINE -> "$1" |"
    print "---------------------------------------------------------------"
    python testing/main.py -i $1 -r $reps | tee -a results/out_$1.log;
fi


    

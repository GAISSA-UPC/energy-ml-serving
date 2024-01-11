#!/bin/bash
# Update package information
#sudo apt-get update

# Upgrade installed packages
#sudo apt-get upgrade -y

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


echo "Initializating server..."
uvicorn app.api_code:app --host 0.0.0.0 --port 8000 > output.log 2>&1 &


SERVER_HOST="localhost"
SERVER_PORT="8000"
CHECK_ENDPOINT="/"  # Adjust to your actual health check endpoint

# Function to check if the server is up
check_server() {
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://${SERVER_HOST}:${SERVER_PORT}${CHECK_ENDPOINT})
    if [ "${HTTP_STATUS}" = "200" ]; then
        echo "Server is UP"
        return 0
    else
        echo "Server is not yet UP (HTTP Status: ${HTTP_STATUS})"
        return 1
    fi
}

# Wait for the server to be up with a timeout
WAIT_TIME=60  # Adjust as needed
SECONDS=0

echo "Checking if the server is up..."
while [ "${SECONDS}" -lt "${WAIT_TIME}" ]; do
    check_server && break
    sleep 1
done

if [ "${SECONDS}" -ge "${WAIT_TIME}" ]; then
    echo "Timeout: Server did not start within ${WAIT_TIME} seconds."
    exit 1
fi

# Continue with your further steps (e.g., run additional commands or scripts)
echo "Server is up. Proceeding with further steps..."
# Add your additional commands here

echo "Running experiments..."
python testing/main.py -i torchscript -r 1


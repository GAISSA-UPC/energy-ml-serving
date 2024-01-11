**ToDo:**

- [x] 

-------------------
# Guide to Deploy ML Models through an API on a Cloud Provider

This guide provides step-by-step instructions to deploy ML models through an API using a virtual machine in a cloud provider. It assumes a general cloud provider setup and covers the following steps:

## Step 1: Choose a Cloud Provider

1. Research and select a cloud provider that best suits for you. Popular options include AWS, Azure, and Google Cloud Platform. In this repo we use the free-tier account from each considered provider. However, this limits the computational resources to work on.

2. Sign up for an account on the chosen cloud provider and follow the instructions to enable the free-tier access.

## Step 2: Provision a Virtual Machine

1. Navigate to the cloud provider's management console.

2. Create a new virtual machine instance with the desired specifications (CPU, RAM, storage, ...). Ensure that you select the free-tier option if available.

3. Set up the virtual machine with the required operating system and configurations.

4. Note down the IP address or hostname of the virtual machine for future reference.

## Step 3: Connect via SSH to VM

1.  Generate your SSH keys 
```shell
ssh-keygen -t rsa -f ~/.ssh/my-key
```
2. Add your public ssh key (my-key.pub) into your cloud provider's allowed keys.
3. Connect to the VM using your private key and the public IP (X.X.X.X) of the VM
```shell
ssh -i ~/.ssh/my-key myuser@X.X.X.X
```
## Step 4: Clone the Repository

1. Install Git on the virtual machine if it's not already installed.

2. Clone the GitHub repository containing the ML model and API code using the following command:

```shell
git clone https://github.com/fjdurlop/deploy-GAISSA.git
```
## Step 5: Set Up the Environment

1. Install the necessary dependencies and packages required for running the API. 

```shell
. ./scripts/setenv.sh
```

```shell

python3 -m pip install -r requirements.txt

```

## Step 6: Run the API

1. Change into the cloned repository directory on the virtual machine.

2. Start the API server:

    ```bash
    uvicorn app.api:app \
        --host 0.0.0.0 \
        --port 5000 \
        --reload \
        --reload-dir app \
        --reload-dir models
    ```
    Or
    ```bash
    uvicorn app.api:app  --host 0.0.0.0 --port 8000  --reload --reload-dir app 
    ```
3. Create SSH tunnel to access endpoints, forwards traffic from my local port XXXX to the remote server's port XXXX
    ```bash
    ssh -L 8000:localhost:8000  myuser@X.X.X.X
    ```
## Step 7: Access the API
1. Open a web browser (http://localhost:8000/) or use a tool like cURL or Postman to make HTTP requests to the API endpoints.

2. Send requests to the API with the required input data and parameters to obtain predictions.


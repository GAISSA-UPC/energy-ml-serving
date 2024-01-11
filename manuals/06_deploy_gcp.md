**ToDo:**

- [x] 

-------------------
# GCP setup: How to deploy ML models in a GCP VM?

1. Get a free GCP account:
   - https://cloud.google.com/?authuser=6&hl=es
2. Go to the [Google Cloud Console](https://console.cloud.google.com/).
3. Create a new project (See 'My First Project').

<center><figure>
  <img
  src="images/gcp/gcp_welcome_screen.png"
</figure></center>
<p style="text-align: center;">Welcome screen.</p>

4. In the left navigation menu, click on "Compute Engine" under the "Compute" section and select "VM instances".

<center><figure>
  <img
  src="images/gcp/gcp_vm_selection.png"
</figure></center>
<p style="text-align: center;">VM instances.</p>

5. Click on the "Create instance" button to create a new virtual machine instance.

<center><figure>
  <img
  src="images/gcp/gcp_create_instance_03.png"
</figure></center>
<p style="text-align: center;">Create VM instance.</p>

6. Select the desired configuration for the virtual machine, such as machine type, CPU, memory, and storage options. Allow https traffic.
- In this case we use: N2 series, 8GB RAM and 30 GB of storage.
- Region: europe-southwest


<center><figure>
  <img
  src="images/gcp/gcp_configure_instance_04.png"
</figure></center>
<p style="text-align: center;">Configure instance.</p>

7. Select an Ubuntu image and 30 GB of storage.

<center><figure>
  <img
  src="images/gcp/gcp_configure_image_05.png"
</figure></center>
<p style="text-align: center;">OS image configuration.</p>

8. Configure the additional settings as per your requirements, including the region, network settings, and SSH access.

9. Review the configuration and click on the "Create" button to create your virtual machine. Note the monthly estimate is \$68 and we have \$300 credit, in that case we will not exceed the free resources. However, you can also set a Budget for your project, and set alarms (See https://cloud.google.com/billing/docs/how-to/budgets).

<center><figure>
  <img
  src="images/gcp/gcp_pricing_07.png"
</figure></center>
<p style="text-align: center;">Pricing.</p>



10. Wait for the virtual machine to be provisioned. Once it is ready, you can find its details on the Compute Engine dashboard, including its public IP.

<center><figure>
  <img
  src="images/gcp/gcp_ip.png"
</figure></center>
<p style="text-align: center;">Get public IP.</p>

11.  To connect to the virtual machine via SSH, click on the SSH button next to the virtual machine instance name. This will open a terminal window directly in your browser.
    
<center><figure>
  <img
  src="images/gcp/gcp_ssh_in_browser.png"
</figure></center>
<p style="text-align: center;">SSH-in-browser</p>

12. You are now connected to your GCP free tier virtual machine and can start using it for your desired tasks.

13. To connect to the VM via SSH from your local machine
    1.  Generate your SSH keys 
    ```shell
        ssh-keygen -t rsa -f ~/.ssh/gcp-vm
    ```
    2. Go to metadata and add your public ssh key
    3. Connect to the VM using your private key
    ```shell
        ssh -i ~/.ssh/gcp-vm fjdur@X.X.X.X
    ```

<center><figure>
  <img
  src="images/gcp/gcp_add_ssh_08.png"
</figure></center>
<p style="text-align: center;">Adding SSH keys.</p>

14.  Clone repository. See [Step 4](03_deploy_general.md)
15.  Set Up the Environment. See [Step 5](03_deploy_general.md)
16.  Run the API. See [Step 6](03_deploy_general.md)
17.  Access the API. See [Step 7](03_deploy_general.md)  
import subprocess
import requests
import json
import threading
import csv
import time
from datetime import datetime



# Global variable to control the loop
running = True
task = 'experiment'

def get_wattmeter_data():
    filename = f'wattmeter_{task}.csv'
    error_log = f'wattmeter_error_log_{task}.txt'
    fieldnames = ['Wattmetter Timestamp', 'True timestamp', 'ID', 'Name', 'State', 'Action', 'Delay',
                  'Current', 'PowerFactor', 'Phase', 'Energy', 'ReverseEnergy', 'EnergyNR', 'ReverseEnergyNR', 'Load']
    url = "http://147.83.72.195/netio.json"
    
    while running:
        with open(filename, mode='a', newline='') as file:
            try:
                response = requests.get(url)
                print(f"response:{response}")
                # Parse JSON data
                data = json.loads(response.text)

                timestamp = data['Agent']['Time']
                print(timestamp)

                # Extract data for output ID 1
                output_1_data = None
                for output in data['Outputs']:
                    if output['ID'] == 1:
                        output_1_data = output
                        break

                if output_1_data:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)

                    if file.tell() == 0:
                        writer.writeheader()

                    output_1_data['Wattmetter Timestamp'] = timestamp
                    output_1_data['True timestamp'] = datetime.fromtimestamp(
                        time.time())
                    writer.writerow(output_1_data)
                else:
                    error_message = f"{datetime.now()} - Output ID 1 data not found in the JSON.\n"
                    with open(error_log, 'a') as error_file:
                        error_file.write(error_message)
            except Exception as e:
                error_message = f"{datetime.now()} - {e}\n"
                with open(error_log, 'a') as error_file:
                    error_file.write(error_message)
            finally: # lets you execute code, regardless of the result of the try-except block
                time.sleep(0.5)  # Pause execution for 0.5 seconds before next call

def profiled_process():
    for i in range(10000):
        print(f"{i}:-")
        #time.sleep(1)
        for i in range(1000): # see the difference of consumed energy, change to 10000 and compare
            print(f"{i}:-")

def main_task():
#if __name__ == "__main__":
    global running
    print(task)

    profiled_process()

    # Set running to False when you want to stop the while loop in get_wattmeter_data()
    running = False

# Inicia la tarea principal en un hilo
if __name__ == "__main__": 
    task_thread = threading.Thread(target=main_task)
    task_thread.start()
    
    get_wattmeter_data()
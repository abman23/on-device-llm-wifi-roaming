"""Context information: location, timestamp.
RSSI values of connected BSSID and highest RSSIs in the past few steps.
CoT prompting, which instructs LLM to generate rationale + answer.

"""
import json
import os
from datetime import datetime


from preprocess.utils import classify_env_and_avail


num_train_timesteps = 10  # number of previous time steps included in prompt

instruction_prompt_template = """Task:
You are provided with 10 time steps of Wi-Fi network context information, including RSSI value of the connected AP, highest RSSI, location and timestamp for each time step. Your goal is to:

Select a reasonable RSSI threshold for Wi-Fi roaming, aiming to strike a balance between connection quality and handover frequency. If the currently connected AP has lower RSSI than your output threshold，we will switch to the AP with the highest RSSI. Otherwise, we will stick to the current one.
Important: RSSI values are negative, so a value closer to zero (e.g., -50) is stronger than one further from zero (e.g., -80).

Guidance:
The network contexts provide crucial information about the dynamics of signal strength of APs:

Connected RSSI: If the connected RSSIs show a consistent decline over time, it may indicate a weakening connection, justifying a higher threshold to trigger a handover proactively. If the connected RSSIs remain stable or fluctuate within a narrow range, a lower threshold is preferred to avoid unnecessary handovers.
Highest RSSI: If the highest RSSIs in the past few steps are significantly better than the connected RSSIs, a higher threshold is appropriate to ensure the UE connects to a stronger AP. If the highest RSSIs are only marginally better or fluctuate significantly, a lower threshold helps prevent frequent handovers.
Location: Movement between locations can impact signal strength. Connected AP with decreasing RSSI as the location changes is moving away from the user, whereas an AP with increasing RSSI may indicate proximity to the user.
Timestamp: Time progression might reveal trends in signal strength.

If the gap between the highest RSSI and the RSSI of the current AP is very large and the context information implies an active handover, you can set a threshold higher than the current RSSI but lower than the highest RSSI to trigger handover. If the gap between current RSSI and highest RSSI is small, or the context information suggests a lower threshold, you can set a threshold lower than the current RSSI to avoid frequent handover.
Dynamically determine the threshold based on the context information, RSSIs of the connected AP, and the highest RSSIs. An appropriate RSSI threshold for roaming triggers is between -67 dBm and -75 dBm.

Network Contexts:
| Time Step | Connected RSSI | Highest RSSI | Location (Longitude, Latitude) | Timestamp |
|-----------|----------------|--------------|------------------------------- |-----------|{ctx_lists} 

Instruction:
Generate a concise response in the following format:

Output Format:
Analysis Summary: [Brief step-by-step explanations of your thinking process, focusing on the reasoning behind choosing certain RSSI thresholds.]
RSSI Threshold: [xx dBm]
"""

completion_prompt_template = "RSSI Threshold: {thr}"


def format_prompts(train_datapoint_list: list[dict]) -> dict:
    """
    Format prompts with the conversational format from the raw data
    :param train_datapoint_list: List of training datapoints ordered by time step
    """
    datapoint = train_datapoint_list[-1]  # current time step
    networks = datapoint["networks"]
    environment, availability, sorted_pairs = classify_env_and_avail(networks, num_threshold=30)
    location = f"({datapoint['location']['longitude']: .6f}, {datapoint['location']['latitude']: .6f})"
    time = datapoint["time"].split(' ')
    timestamp = time[0] + 'T' + time[1]
    battery_info = datapoint["battery_info"]
    capacity = battery_info["Current Capacity"]

    return {"prompt": instruction_prompt_template, "sorted_pairs": sorted_pairs,
            'location': location, "timestamp": timestamp, "capacity": capacity}


def format_dataset(filepath_list: list[str], save: bool = True,
                   reasoning_filepath_list: list[str] = None):
    """
    Format the prompt strings from the original dataset
    :param filepath_list: paths to the collections of original data
    :param save: Save the formatted prompts as a json file or not
    :param reasoning_filepath_list: List of paths to the reasoning corpora
    :return: List of formatted prompts
    """
    prompt_list = []

    for file_idx, filepath in enumerate(filepath_list):
        datapoints = []
        with open(filepath, "r") as file:
            for num, line in enumerate(file):
                datapoint = json.loads(line.strip())

                # Data cleaning
                if 'networks' not in datapoint or not datapoint['networks']:
                    continue
                datapoints.append(datapoint)

            for i in range(num_train_timesteps, len(datapoints)):
                datapoint_window = datapoints[i - num_train_timesteps: i]
                prompt = format_prompts(train_datapoint_list=datapoint_window)
                prompt_list.append(prompt)

            if reasoning_filepath_list is not None:
                # Add the reasoning as label to the message lists
                reasoning_filepath = reasoning_filepath_list[file_idx]
                with open(reasoning_filepath, "r") as reasoning_file:
                    for num, line in enumerate(reasoning_file):
                        reasoning = json.loads(line.strip())
                        prompt_list[num]['messages'].append(reasoning)  # add the reasoning as label

    if save:
        saved_filename = datetime.now().strftime(f"task2_%m-%d-%H-%M.json")
        data_filedir = '../formatted_data'
        if not os.path.exists(data_filedir):
            os.makedirs(data_filedir)

        saved_filepath = os.path.join(data_filedir, saved_filename)
        with open(saved_filepath, 'a') as file:
            for prompt in prompt_list:
                # Exclude fields other than the prompt text when saving
                prompt = {"messages": prompt['messages']}
                json.dump(prompt, file)
                file.write('\n')

    return prompt_list

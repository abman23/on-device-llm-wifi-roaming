"""Context input: location, timestamp.
0-shot CoT prompting, which instructs LLM to generate rationale + answer.

"""
import json
import os
from datetime import datetime


from preprocess.utils import find_stable_bssid, weak_rssi_thr

num_train_timesteps = 10  # number of previous timesteps used as reference for LLM
num_eval_timesteps = 100  # number of future timesteps used for counting the consecutive stable BSSIDs
num_bssid_limit = 10  # max number of BSSID-RSSI pairs at one timestep (choose the BSSIDs with the N largest RSSIs)

instruction_prompt_template = """Task:
You are provided with {num_step} time steps of BSSID lists, each including the RSSI (Received Signal Strength Indicator) values for different BSSIDs. You are also given context information including location (longitude, latitude) and timestamp. Your goal is to:
 
Select a BSSID from the list at the last time step ({num_step}) that is most likely to maintain an RSSI above {thr} for the longest consecutive future time steps.
Important: RSSI values are negative, so a value closer to zero (e.g., -50) is stronger than one further from zero (e.g., -80).

Guidance:
The location and timestamp at each time step provide crucial information about the dynamics of BSSID signal strength and the appropriate handover decision:

Location: Movement between locations can impact signal strength. A BSSID with decreasing RSSI as the location changes is moving away from the user, whereas a BSSID with increasing RSSI may indicate proximity to the user. The optimal BSSID at one location may still be the optimal one when you see it next time.
Timestamp: Time progression might reveal trends in signal strength. A BSSID with steady or increasing RSSI could indicate stability, while one with frequent fluctuations may be less reliable in future steps.

Analyze how RSSI values change with respect to location and timestamp to determine which BSSID is likely to maintain a strong signal.

BSSID Lists and Contexts:
| Time Step | BSSID / RSSI list | Location (Longitude, Latitude) | Timestamp |
|-----------|-------------------|--------------------------------|---------- |{bssid_lists}	

Instruction:
Analyze the changes in RSSI values with respect to location and timestamp:

Avoid BSSIDs with decreasing RSSI as location shifts, or timestamp advances.
Prioritize BSSIDs with stable or increasing RSSI, as it may support stronger and more consistent signal reception.

Generate a concise response in the following format:
Analysis Summary: Briefly explain the selection process, focusing on the reasoning behind excluding or prioritizing certain BSSIDs.
Chosen BSSID: Provide the selected BSSID at the end in this format: Chosen BSSID: xx:xx:xx:xx:xx:xx
Important: Do not include lengthy illustrations.

Output Format:
Analysis Summary: [Brief step-by-step explanations]
Chosen BSSID: xx:xx:xx:xx:xx:xx
"""

completion_prompt_template = "Chosen BSSID: {chosen_bssids}"


def format_prompts(train_datapoint_list: list[dict], eval_networks_window: list[list[dict]], switch_rssi_thr: int) -> dict:
    """
    Format prompts with the conversational format from the raw data
    :param train_datapoint_list: List of training datapoints ordered by time step
    :param eval_networks_window: List of networks data used for evaluation
    """
    conversations = []
    bssid_lists = ''

    final_pairs = []
    for i, datapoint in enumerate(train_datapoint_list):
        networks = datapoint["networks"]
        time = datapoint["time"].split(' ')
        timestamp = time[0] + 'T' + time[1]
        location = f"({datapoint['location']['longitude']: .6f}, {datapoint['location']['latitude']: .6f})"
        pairs = {}
        for idx, network in enumerate(networks):
            rssi = network["rssi"]
            if rssi <= weak_rssi_thr:
                continue
            bssid = network["bssid"]
            pairs[bssid] = rssi

        pairs_sorted = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
        final_pairs = pairs_sorted
        if len(pairs_sorted) > num_bssid_limit:
            # for the token limit of LLM
            pairs_sorted = pairs_sorted[:num_bssid_limit]
        bssid_list = f'| {i + 1} |' + ' {' + ', '.join(
            [f'{bssid} / {rssi}' for (bssid, rssi) in pairs_sorted]) + '} | ' + f'{location} | {timestamp} |'
        bssid_lists += f'\n{bssid_list}'

    instruction_prompt = instruction_prompt_template.format(num_step=num_train_timesteps, thr=switch_rssi_thr,
                                                            bssid_lists=bssid_lists)
    chosen_bssids, step_counts, rssi_sum = find_stable_bssid(networks_window=eval_networks_window,
                                                             rssi_threshold=switch_rssi_thr)
    completion_prompt = completion_prompt_template.format(chosen_bssids=str(chosen_bssids[0]))

    conversations.append({"role": "user", "content": instruction_prompt})
    conversations.append({"role": "assistant", "content": completion_prompt})

    return {'messages': conversations, "completion": chosen_bssids, "counts": step_counts, "rssi_sum": rssi_sum, "pairs": final_pairs}


def format_dataset(filepath_list: list[str], switch_rssi_thr: int, save: bool = True, reasoning_filepath_list: list[str] = None):
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
            rssi_list = []
            for num, line in enumerate(file):
                datapoint = json.loads(line.strip())

                # Data cleaning
                if 'networks' not in datapoint or not datapoint['networks']:
                    continue
                networks = datapoint["networks"]
                rssi_max = -100
                rssis = []
                for network in networks:
                    rssi = network["rssi"]
                    rssis.append(rssi)
                    # filter out BSSID with low RSSI
                    if rssi > rssi_max:
                        rssi_max = rssi
                if rssi_max < switch_rssi_thr:
                    continue
                rssi_list.append(rssis)

                datapoints.append(datapoint)

            for i in range(num_train_timesteps, len(datapoints) - num_eval_timesteps):
                datapoint_window = datapoints[i - num_train_timesteps: i]
                # the next {num_eval_timesteps} timesteps (including the current tiemstep)
                eval_networks_window = [datapoint['networks'] for datapoint in
                                        datapoints[i - 1: i + num_eval_timesteps - 1]]
                prompt = format_prompts(train_datapoint_list=datapoint_window,
                                        eval_networks_window=eval_networks_window, switch_rssi_thr=switch_rssi_thr)
                prompt_list.append(prompt)

            if reasoning_filepath_list is not None:
                # Add the reasoning as label to the message lists
                reasoning_filepath = reasoning_filepath_list[file_idx]
                with open(reasoning_filepath, "r") as reasoning_file:
                    for num, line in enumerate(reasoning_file):
                        reasoning = json.loads(line.strip())
                        prompt_list[num]['messages'][-1] = reasoning  # replace BSSID by reasoning + BSSID

    if save:
        saved_filename = datetime.now().strftime(f"task1_data_%m-%d-%H-%M.json")
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

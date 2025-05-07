"""Inference script for context-aware AP choice.

"""
import copy
import os
import random
import re
import time
import argparse

import numpy as np
import ollama

from preprocess.preprocess_task1 import format_dataset


def main(switch_rssi_thr, data_file_path, model):
    n_examples_avg, n_ho_avg, n_ho_opt_avg, n_qa_avg, n_correct_avg, n_ho_rand_avg = [], [], [], [], [], []
    rssi_avg, rssi_rand_avg, rssi_opt_avg = [], [], []
    n_ho_greedy_avg, rssi_greedy_avg = [], []
    inf_time = []
    data_directory_path = 'data'
    # for filename in os.listdir(data_directory_path):
    data_files = [data_file_path] if isinstance(data_file_path, str) else data_file_path
    switch_thr = [switch_rssi_thr] if isinstance(switch_rssi_thr, int) else switch_rssi_thr

    start = time.time()
    for filename, thr in zip(data_files, switch_thr):
        print(f"**Dataset**: {filename}")
        # Construct full file path of each raw data file
        data_file_path = os.path.join(data_directory_path, filename)
        # Prepare the evaluation dataset, which includes a time series of network data
        example_list = format_dataset([data_file_path], switch_rssi_thr=thr, save=False)
        n_examples = len(example_list)
        print(f"Data preprocessing finished. {n_examples} examples\n")

        # Evaluation (count the number of handover actions in the time series)
        n_qa, n_correct = 0, 0
        step = 0
        n_ho = 0
        sum_rssi, sum_rssi_rand, sum_rssi_opt = 0, 0, 0
        sum_rssi_greedy = 0
        repeat = 0
        response_history = ''
        while step < n_examples:
            print(f"**Step**: {step}, {n_ho} handover actions and {n_qa} inferences so far.")
            example = example_list[step]
            messages = copy.deepcopy(example['messages'])[:-1]  # no answer should be presented
            completion = example['completion']
            counts = example['counts']
            rssi_table = example['rssi_sum']
            max_count = counts[completion[0]]

            # LLM inference through API
            if repeat > 0:
                messages.append({"role": "assistant", "content": response_history})
                messages.append({"role": "user",
                                 "content": f"Please reconsider the situation and return another BSSID."})
            if repeat < 5:
                # 5 chances for the LLM agent
                inf_start = time.time()
                response = ollama.chat(model=model, messages=messages,
                                       options={'num_ctx': 5000})
                inf_end = time.time()
                inf_time.append(inf_end - inf_start)
                text = response['message']['content']
            else:
                # in case LLM agent gets in a dead loop
                print("LLM agent is replaced by random selection")
                valid_bssid_list = [bssid for bssid, cnt in counts.items() if cnt > 0]
                text = random.choice(valid_bssid_list)
            n_qa += 1

            # Extract BSSIDs from the outputs and calculate the accuracy
            pattern = r'(?:[0-9A-Fa-f]{2}[:.-]){5}[0-9A-Fa-f]{2}'

            matches = re.findall(pattern, text)
            if matches:
                bssid = matches[-1]

                chosen_count = counts.get(bssid, 0)
                rssi = rssi_table.get(bssid, 0)
                if chosen_count > 0:
                    print("**Results**:")
                    print(f"Chosen BSSID {bssid} moves {chosen_count} time steps forward, "
                          f"while the optimal BSSID {completion[0]} will move {max_count} steps forward.\n")
                    if chosen_count == max_count:
                        n_correct += 1
                    n_ho += 1
                    sum_rssi += rssi
                    step += chosen_count
                    repeat = 0
                    continue
            # handover again if no valid response
            repeat += 1
            response_history = text

        # Calculate the minimum number of handover actions
        step_opt = 0
        n_ho_opt = 0
        while step_opt < n_examples:
            example = example_list[step_opt]
            completion = example['completion']
            counts = example['counts']
            rssi_table = example['rssi_sum']
            max_count = counts[completion[0]]
            step_opt += max_count
            sum_rssi_opt += rssi_table.get(completion[0], 0)
            n_ho_opt += 1

        # Evaluate random selection
        step_rand = 0
        n_ho_rand = 0
        while step_rand < n_examples:
            example = example_list[step_rand]
            counts = example['counts']
            rssi_table = example['rssi_sum']
            valid_bssid_list = [bssid for bssid, cnt in counts.items() if cnt > 0]
            rand_bssid = random.choice(valid_bssid_list)
            rand_count = counts[rand_bssid]
            sum_rssi_rand += rssi_table.get(rand_bssid, 0)
            step_rand += rand_count
            n_ho_rand += 1

        # Evaluate greedy selection
        step_greedy = 0
        n_ho_greedy = 0
        while step_greedy < n_examples:
            example = example_list[step_greedy]
            counts = example['counts']
            rssi_table = example['rssi_sum']
            pairs = example['pairs']
            greedy_bssid = pairs[0][0]
            greedy_count = counts[greedy_bssid]
            sum_rssi_greedy += rssi_table.get(greedy_bssid, 0)
            step_greedy += greedy_count
            n_ho_greedy += 1

        n_examples_avg.append(n_examples)
        n_ho_avg.append(n_ho)
        n_ho_opt_avg.append(n_ho_opt)
        n_qa_avg.append(n_qa)
        n_correct_avg.append(n_correct)
        n_ho_rand_avg.append(n_ho_rand)
        n_ho_greedy_avg.append(n_ho_greedy)
        rssi_avg.append(sum_rssi / step)
        rssi_rand_avg.append(sum_rssi_rand / step_rand)
        rssi_opt_avg.append(sum_rssi_opt / step_opt)
        rssi_greedy_avg.append(sum_rssi_greedy / step_greedy)
        print(f"\n***** Dataset {filename} Statistics *****")
        print(
            f"Time series length: {n_examples};\n"
            f"minimum number of handover actions: {n_ho_opt};\n"
            f"number of handover actions done by traditional method: {n_ho_rand};\n"
            f"number of handover actions done by LLM agent: {n_ho};\n"
            f"number of inferences made by LLM agent: {n_qa};\n"
            f"number of optimal handover decisions made by LLM agent: {n_correct}.\n"
        )

    print("\n********** Summary **********")
    print(f"Data file path: {data_file_path}, model name: {model}")
    print(
        f"Evaluation datasets: {data_files};\n"
        f"Totally {len(n_examples_avg)} evaluation time series;\n"
        f"average time series length: {np.mean(n_examples_avg)};\n"
        f"average minimum number of handover actions: {np.mean(n_ho_opt_avg)};\n"
        f"average number of handover actions done by random selection: {np.mean(n_ho_rand_avg)};\n"
        f"average number of handover actions done by greedy method: {np.mean(n_ho_greedy_avg)};\n"
        f"average number of handover actions done by LLM agent: {np.mean(n_ho_avg)};\n"
        f"average number of inferences made by LLM agent: {np.mean(n_qa_avg)};\n"
        f"average number of optimal handover decisions made by LLM agent: {np.mean(n_correct_avg)}.\n"
        f"average accuracy of making the optimal handover: {np.sum(n_correct_avg) / np.sum(n_ho_avg)}.\n"
        f"average ratio of valid handover: {np.sum(n_ho_avg) / np.sum(n_qa_avg)}.\n"
        f"average RSSI of BSSIDs selected by LLM agent: {np.mean(rssi_avg)};\n"
        f"average RSSI of random selected BSSIDs: {np.mean(rssi_rand_avg)};\n"
        f"average RSSI of greedy selected BSSIDs: {np.mean(rssi_greedy_avg)};\n"
        f"average RSSI of optimal BSSIDs: {np.mean(rssi_opt_avg)};\n"
    )

    print(f"Average inference time: {np.mean(inf_time) :.4f} s for {len(inf_time)} inferences.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for the script.")

    # Define the command-line arguments
    parser.add_argument("--thr", type=int, nargs='+', dest="switch_rssi_thr", default=-70,
                        help="Set the switch RSSI threshold (e.g., -70)")
    parser.add_argument("--data", type=str, nargs='+', default='hybrid_2.json',
                        help="Set the data file path (e.g., xxx.json)")
    parser.add_argument("--model", type=str,
                        help="Set the model name as a path or a hf model name")

    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the main function
    main(args.switch_rssi_thr, args.data, args.model)

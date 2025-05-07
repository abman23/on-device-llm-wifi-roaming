"""Inference script for online roaming optimization.

"""
import os
import re
import time
import argparse

import numpy as np
import ollama

from preprocess.preprocess_task2 import format_dataset, num_train_timesteps


def main(fixed_thr, data_file_path, interval, model):
    fixed_thr = [fixed_thr] if isinstance(fixed_thr, int) else fixed_thr
    n_fixed_thr = len(fixed_thr)
    n_examples_avg, n_ho_avg, n_ho_highest_avg, n_qa_avg, n_valid_avg, n_ho_fixed_avg = [], [], [], [], [], [[] for _ in range(n_fixed_thr)]
    rssi_avg, rssi_fixed_avg, rssi_highest_avg = [], [[] for _ in range(n_fixed_thr)], []
    thr_avg = []
    inf_time = []
    data_directory_path = 'data'
    # for filename in os.listdir(data_directory_path):
    data_files = [data_file_path] if isinstance(data_file_path, str) else data_file_path
    # data_files = ['data_06-19-17-11_b2.json', 'data_06-20-11-08_b1.json']
    start = time.time()
    for filename in data_files:
        print(f"**Dataset**: {filename}")
        # Construct full file path of each raw data file
        data_file_path = os.path.join(data_directory_path, filename)
        # Prepare the evaluation dataset, which includes a time series of network data
        example_list = format_dataset([data_file_path], save=False)
        n_examples = len(example_list)
        print(f"Data preprocessing finished. {n_examples} examples\n")

        # Evaluation (count the number of handover actions in the time series)
        n_qa, n_valid = 0, 0
        step = interval
        n_ho, n_ho_highest, n_ho_fixed = 0, 0, [0] * n_fixed_thr,
        sum_rssi, sum_rssi_fixed, sum_rssi_highest = [], [[] for _ in range(n_fixed_thr)], []
        thresholds = []
        init_pairs = example_list[0]['sorted_pairs']
        connected_bssid = list(init_pairs.keys())[0]

        while step <= n_examples:
            print(f"**Step**: {step}, {n_ho} handover actions and {n_qa} inferences so far.")
            examples = example_list[step - interval: step]
            example = examples[-1]
            ctx_lists = ''
            for i, e in enumerate(examples[-1:-num_train_timesteps:-1]):
                pairs = e['sorted_pairs']
                location = e['location']
                timestamp = e['timestamp']
                ctx_list = f"| {i + 1} | {pairs.get(connected_bssid, -100)} | {list(pairs.values())[0]} | {location} | {timestamp} |"
                ctx_lists += f'\n{ctx_list}'
            prompt = example['prompt'].format(ctx_lists=ctx_lists)

            step += interval
            sorted_pairs = example['sorted_pairs']
            bssid_list = list(sorted_pairs.keys())
            connected_rssi = sorted_pairs.get(connected_bssid, -100)
            messages = []
            messages.append({'role': 'user',
                             'content': prompt})

            # LLM inference through Ollama API
            inf_start = time.time()
            response = ollama.chat(model=model, messages=messages, options={'num_ctx': 2048})
            inf_end = time.time()
            inf_time.append(inf_end - inf_start)
            text = response['message']['content']
            # print(f"**Prompt**:\n{messages[-1]['content']}")
            # print(f"**Response**:\n{text}")
            n_qa += 1

            # Extract threshold from the outputs
            pattern = r'-\d+'  # negative integer

            matches = re.findall(pattern, text)
            if matches:
                thr = int(matches[-1])
                thresholds.append(thr)
                n_valid += 1
                print(f"**Results**: RSSI threshold: {thr}")
            else:
                print("**Warning**: No threshold returned. Use the default threshold -70.")
                thr = -70

            future_examples = example_list[step - interval: min(step, len(example_list))]
            for e in future_examples:
                pairs = e['sorted_pairs']
                rssi = pairs.get(connected_bssid, -100)
                highest_bssid = list(pairs.keys())[0]
                highest_rssi = pairs[highest_bssid]
                if rssi < thr and rssi < highest_rssi:
                    # handover if current RSSI lower than threshold
                    connected_bssid = highest_bssid
                    rssi = highest_rssi
                    n_ho += 1
                sum_rssi.append(rssi)

        # baseline - always choose the bssid with the highest rssi
        step = 0
        connected_bssid = list(init_pairs.keys())[0]
        while step < n_examples:
            example = example_list[step]
            sorted_pairs = example['sorted_pairs']
            bssid_list = list(sorted_pairs.keys())
            step += 1

            highest_bssid = bssid_list[0]
            highest_rssi = sorted_pairs[highest_bssid]
            if connected_bssid != highest_bssid:
                n_ho_highest += 1
                connected_bssid = highest_bssid
            sum_rssi_highest.append(highest_rssi)

        # baseline - fixed threshold
        for i, thr in enumerate(fixed_thr):
            step = 0
            connected_bssid = list(init_pairs.keys())[0]
            while step < n_examples:
                example = example_list[step]
                sorted_pairs = example['sorted_pairs']
                bssid_list = list(sorted_pairs.keys())
                step += 1
                connected_rssi = sorted_pairs.get(connected_bssid, -100)

                if connected_rssi < thr:
                    connected_bssid = bssid_list[0]
                    fixed_rssi = sorted_pairs[connected_bssid]
                    n_ho_fixed[i] += 1
                else:
                    fixed_rssi = connected_rssi
                sum_rssi_fixed[i].append(fixed_rssi)

        n_examples_avg.append(n_examples)
        n_ho_avg.append(n_ho)
        n_ho_highest_avg.append(n_ho_highest)
        n_qa_avg.append(n_qa)
        n_valid_avg.append(n_valid)
        rssi_avg.append(np.mean(sum_rssi))
        for i in range(n_fixed_thr):
            rssi_fixed_avg[i].append(np.mean(sum_rssi_fixed[i]))
            n_ho_fixed_avg[i].append(n_ho_fixed[i])
        rssi_highest_avg.append(np.mean(sum_rssi_highest))
        thr_avg.append(thresholds)
        print(f"\n***** Dataset {filename} Statistics *****")
        print(
            f"Time series length: {n_examples};\n"
            f"number of handover with dynamic threshold: {n_ho};\n"
            f"average RSSI with dynamic threshold: {np.mean(sum_rssi)};\n"
            f"number of inferences made by LLM agent: {n_qa};\n"
            f"number of valid threshold returned by LLM agent: {n_valid}.\n"
        )
        for i, thr in enumerate(fixed_thr):
            print(f"number of handover with fixed threshold {thr}: {n_ho_fixed[i]};")
        print(f"number of handover with greedy selection of the highest RSSI: {n_ho_highest};")

    print("\n********** Summary **********")
    print(f"Model name: {model}, inference interval: {interval}")
    print(
        f"Evaluation datasets: {data_files};\n"
        f"Totally {len(n_examples_avg)} evaluation time series;\n"
        f"average time series length: {np.mean(n_examples_avg)};\n"
        f"average number of inferences made by LLM agent: {np.mean(n_qa_avg)};\n"
        f"average ratio of valid inference: {np.sum(n_valid_avg) / np.sum(n_qa_avg)}.\n"
        f"average dynamic threshold: {[np.mean(thr) for thr in thr_avg]}, std: {[np.std(thr) for thr in thr_avg]};\n"
        f"average number of handover with dynamic threshold: {np.mean(n_ho_avg)};"
    )
    for i, thr in enumerate(fixed_thr):
        print(f"average number of handover with fixed threshold {thr}: {np.mean(n_ho_fixed_avg[i])};")
    print(f"average number of handover with greedy selection of the highest RSSI: {np.mean(n_ho_highest_avg)};")
    print(f"average RSSI with dynamic threshold: {np.mean(rssi_avg)};")
    for i, thr in enumerate(fixed_thr):
        print(f"average RSSI with fixed threshold {thr}: {np.mean(rssi_fixed_avg[i])};")
    print(f"average RSSI with greedy selection of the highest RSSI: {np.mean(rssi_highest_avg)};")
    avg_num_ho = np.mean(n_ho_avg)
    avg_rssi = np.mean(rssi_avg)
    hybrid_metric = avg_num_ho - 10 * avg_rssi
    print(f"hybrid metric: {hybrid_metric:.4f}")

    print(f"{round(time.time() - start, 2)} seconds used for evaluation.")
    print(f"Average ollama inference time: {np.mean(inf_time) :.4f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters for the script.")

    # Define the command-line arguments
    parser.add_argument("--thr", type=int, nargs='+', default=-70,
                        help="Set the fixed switch RSSI threshold (e.g., -70)")
    parser.add_argument("--data", type=str, nargs='+', default='hybrid_2.json',
                        help="Set the data file paths (e.g., xxx.json)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Set the LLM inference interval")
    parser.add_argument("--model", type=str,
                        help="Set the Ollama model tag")

    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the main function
    main(fixed_thr=args.thr, data_file_path=args.data,
         interval=args.interval, model=args.model)

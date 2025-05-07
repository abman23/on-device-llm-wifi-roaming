import copy
import random
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2


# Haversine formula to calculate distance between two points
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# Calculate speeds and classify mobility
def classify_mobility(locations, threshold=1.5):  # threshold in m/s
    d, t = 0, 0
    for i in range(1, len(locations)):  # lon, lat, time (seconds)
        lon1, lat1, t1 = locations[i - 1]
        lon2, lat2, t2 = locations[i]
        distance = haversine(lon1, lat1, lon2, lat2)
        time_diff = t2 - t1
        d += distance
        t += time_diff

    # Average speed
    avg_speed = d / t if t != 0 else 0
    # print(f"avg_speed: {avg_speed}")
    mobility = "high" if avg_speed > threshold else "low"
    return mobility, avg_speed


def classify_hours(timestamp: str, scenario: str = 'office'):
    # Parse the timestamp string into a datetime object
    dt_object = datetime.strptime(timestamp, "%H:%M:%S")

    # Extract the hour
    hour = dt_object.hour
    t = dt_object.strftime("%H:%M")

    # Determine if it's peak or off-peak
    if 9 <= hour < 12 or 13 <= hour < 18:
        return "peak", t
    else:
        return "off-peak", t


def classify_battery(capacity: int):
    return "high" if capacity >= 30 else "low"


def classify_env_and_avail(networks: list, num_threshold: int = 10):
    """Determine environment scenario and BSSID availability

    :param networks:
    :param num_threshold:
    :return: (env, avail, sorted_pairs)
    """
    pairs = {}
    connected_bssid = ''
    connected_rssi = -100
    for network in networks:
        rssi = network["rssi"]
        bssid = network["bssid"]
        if network['noise'] != 0:
            connected_bssid = bssid
            connected_rssi = rssi
        if rssi < weak_rssi_thr:
            continue

        pairs[bssid] = rssi

    pairs_sorted = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
    k = min(9, len(pairs_sorted) - 1)
    rssi_top10 = pairs_sorted[k][1]
    if abs(rssi_top10 + 70) < abs(rssi_top10 + 60):
        env = 'outdoor'
    else:
        env = 'indoor'

    avail = "high" if len(pairs) >= num_threshold else "low"
    # print(f"n_bssids: {len(pairs)}")

    return env, avail, dict(pairs_sorted)


weak_rssi_thr = -90  # lower bound for RSSI to be considered


def find_stable_bssid(networks_window: list[list[dict]], rssi_threshold: int,
                      thr_list=None) -> tuple[list[str], dict, dict]:
    """
    Find BSSIDs that keep their RSSI values above a threshold for the most consecutive time steps starting from 0.
    :param thr_list: a list of RSSI thresholds in case you need different thresholds for the BSSID-RSSI pair in each time step.
    :param networks_window: networks lists ordered by timestep
    :param rssi_threshold: RSSI threshold
    :return: A list of the most stable BSSIDs, the counts of consecutive time steps of each initial BSSID, and their
    accumulated RSSI in the time window
    """
    # Dictionary to keep track of the maximum consecutive count for each BSSID
    consecutive_counts: dict[str: int] = {}
    accumulated_rssi: dict[str, int] = {}

    for timestep, networks in enumerate(networks_window):
        for network in networks:
            bssid, rssi = network["bssid"], network["rssi"]
            if rssi <= weak_rssi_thr:
                continue
            if timestep == 0:  # Initialize all BSSIDs at timestep 0
                consecutive_counts[bssid] = 0
                accumulated_rssi[bssid] = 0

            thr = rssi_threshold if thr_list is None else thr_list[timestep]
            if rssi >= thr:
                if timestep == consecutive_counts.get(bssid, 0):
                    consecutive_counts[bssid] += 1
                    accumulated_rssi[bssid] += rssi
            # if timestep == 0:  # Initialize all BSSIDs at timestep 0
            #     consecutive_counts[bssid] = 1
            #     accumulated_rssi[bssid] = rssi
            # else:
            #     thr = rssi_threshold if thr_list is None else thr_list[timestep]
            #     if rssi >= thr:
            #         if timestep == consecutive_counts.get(bssid, 0):
            #             consecutive_counts[bssid] += 1
            #             accumulated_rssi[bssid] += rssi

        # print(consecutive_counts)

    # Finding the BSSID with the maximum consecutive count
    max_count = 0
    stable_bssids = []
    # print('Counts:\n')
    # print(consecutive_counts)

    for bssid, count in consecutive_counts.items():
        if count > max_count:
            max_count = count
            stable_bssids = [bssid]
        elif count == max_count:
            stable_bssids.append(bssid)

    return stable_bssids, consecutive_counts, accumulated_rssi


def format_preference_data(original_data_list, max_number_per_data=10):
    """Format a dataset for DPO where each row is a (prompt, chosen, rejected) tuple.

    :param max_number_per_data: max number of generated data for one original data.
    :param original_data_list: list of preprocessed data points from 'preprocess_v43'.
    :return: list of tuples used for DPO trainer.
    """
    formatted_data_list = []
    for data in original_data_list:
        prompt = data['messages'][:1]
        chosen_bssids = data['completion']
        chosen, rejected = [], []
        chosen_cnt, rejected_cnt = 0, 0
        for bssid, cnt in data['counts'].items():
            if bssid in chosen_bssids and chosen_cnt < max_number_per_data:
                chosen.append([{'role': 'assistant', 'content': f'Chosen BSSID: {bssid}'}])
                chosen_cnt += 1
            elif bssid not in chosen_bssids and rejected_cnt < max_number_per_data:
                rejected.append([{'role': 'assistant', 'content': f'Chosen BSSID: {bssid}'}])
                rejected_cnt += 1

        n = min(len(chosen), len(rejected), max_number_per_data)
        pairs = list(zip(random.sample(chosen, n), random.sample(rejected, n)))
        for chosen_msg, rejected_msg in pairs:
            formatted_data_list.append({'prompt': copy.deepcopy(prompt), 'chosen': copy.deepcopy(chosen_msg), 'rejected': copy.deepcopy(rejected_msg)})

    return formatted_data_list

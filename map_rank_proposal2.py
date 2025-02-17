import json
import pathlib
import numpy as np
import datetime 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

def sanitize_name_md(name): return name.replace('|', r'\|')

def read_map_file(map_name, pro = False):
    path = "maptops-pro" if pro else "maptops"
    try:
        with open(f"{path}/{map_name}.json", "r", encoding= 'utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file '{map_name}.json' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

def build_scorer(wr_time, median_time, completions):
    wr = wr_time
    v = 0.2 # controls median score as completions -> inf
    q = 1/3 # controls floor value (affects how flat the curve is)
    u = 1 / np.power(completions, q) # floor value
    t = v * (1 - u) + u # score at median time
    k = np.log((1 - u + 1e-6) / (t - u)) / median_time # coefficient to respect median condition

    def scorer(time):
        diff = time - wr
        return int(10000 * (u + (1 - u)/np.exp(k * diff)))
    
    return scorer

def create_markdown_table(map_name, headers, table):
    with open(f"tables2/{map_name}.md", "w", encoding="utf-8") as f:
        f.write(f"# {map_name} Maptop\n\n")
        f.write("| ")
        for h in headers:
            f.write(f" {h} |")
        f.write("\n")
        f.write("| ")
        for _ in range(len(headers)):
            f.write("-------------- |")
        f.write("\n")
        for i in range(len(table[0])): # row idx
            f.write("| ")
            for c in table:
                f.write(f" {c[i]} |")
            f.write("\n")

def parse(map_name):
    raw_json_data = read_map_file(map_name)
    if not raw_json_data: 
        return
    json_data = sorted(raw_json_data, key=lambda x: x["time"])
    wr_time = json_data[0]["time"]
    median_time = json_data[len(json_data)//2]['time']
    scorer = build_scorer(wr_time, median_time, len(json_data))
    table_headers = ["#", "Name", "Time", "Points"]
    table = [[], [], [], []] # name, time, points
    for i, data in enumerate(json_data):
        table[0].append(i+1)
        table[1].append(sanitize_name_md(data['player_name']))
        table[2].append(str(datetime.timedelta(seconds=data['time'])))
        table[3].append(scorer(data['time']))
    create_markdown_table(map_name, table_headers, table)

def main():
    path = "maptops"
    maps = [f.stem for f in pathlib.Path(path).iterdir()]
    # maps = ["kz_kiwiterror"]

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(parse, maps), total=len(maps)))

if __name__ == "__main__":
    main()
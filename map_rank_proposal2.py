import json
import pathlib
import numpy as np
import datetime 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

def sanitize_name_md(name): return name.replace('|', r'\|')

def read_map_file(dir_name, map_name, pro = False):
    try:
        with open(f"{dir_name}/{map_name}.json", "r", encoding= 'utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file '{map_name}.json' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

# https://www.desmos.com/calculator/cugh0ahivb
def build_scorer(wr_time, sample_time, completions):
    wr = wr_time
    v = 0.2 # controls median score as completions -> inf
    q = 1/3 # controls floor value (affects how flat the curve is)
    p = 1 # controls curviness before sample time (higher values mean less difference at the top times)
    u = 1 / np.power(completions, q) # floor value
    t = v * (1 - u) + u # score at sample time
    k = np.log((1 - u + 1e-6) / (t - u)) / np.power(sample_time, p) # coefficient to respect sample condition

    def scorer(time):
        diff = time - wr
        return int(10000 * (u + (1 - u)/np.exp(k * np.power(diff, p))))
    
    return scorer

def sampler(completions):
    m = 0.4 # percentile asymptote
    q = 1/3 # percentile variation power
    u = 1 / np.power(completions, q)
    return int(completions * (u + (1 - u)*m)) # converges to 100*m percentile value as completions -> inf

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

def create_plots(map_name, p0, p1):
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        fig.suptitle(map_name, fontsize=16)
        
        # Time distribution
        ax1.hist(p0[0], bins=200, edgecolor='black')
        ax1.set_title('Time Distribution')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Count')
        
        # Score distribution
        # ax2.hist(table[1], bins=200, edgecolor='black')
        # ax2.set_title('Score Distribution')
        # ax2.set_xlabel('Score')
        # ax2.set_ylabel('Count')
        # ax2.invert_xaxis()

        cumulative0 = np.arange(1, len(p0[1]) + 1) / len(p0[1])  # Normalized cumulative counts
        cumulative1 = np.arange(1, len(p1[1]) + 1) / len(p1[1])

        ax2.plot(p0[1], cumulative0, '-', label="praetor")
        ax2.plot(p1[1], cumulative1, '-', label="zerok")
        ax2.set_title('Cumulative Score Distribution')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Cumulative Proportion')
        ax2.invert_xaxis()
        ax2.grid(True)
        ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax2.xaxis.set_major_locator(plt.MultipleLocator(1000))
        ax2.legend()
        
        plt.tight_layout()
        
        # Save and close the figure (important for memory management in parallel processing)
        plt.savefig(f"plots/{map_name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating plot for {map_name}: {e}")

def parse(map_name):
    raw_json_data = read_map_file("maptops", map_name)
    if not raw_json_data: 
        return
    json_data = sorted(raw_json_data, key=lambda x: x["time"])
    wr_time = json_data[0]["time"]
    # median_time = json_data[len(json_data)//2]['time']
    median_time = json_data[sampler(len(json_data))]['time']
    scorer = build_scorer(wr_time, median_time, len(json_data))
    table_headers = ["#", "Name", "Time", "Points"]
    table = [[], [], [], []] # name, time, points
    table2 = [[], []]
    for i, data in enumerate(json_data):
        table[0].append(i+1)
        table[1].append(sanitize_name_md(data['player_name']))
        table[2].append(str(datetime.timedelta(seconds=data['time'])))
        table[3].append(scorer(data['time']))
        table2[0].append(data['time'])
        table2[1].append(scorer(data['time']))
    create_markdown_table(map_name, table_headers, table)

    # Comparison with current proposal
    raw_json_data_proposal = read_map_file("maptop-proposal", map_name + "-NUB")
    json_data_proposal = sorted(raw_json_data_proposal, key=lambda x: x["time"])
    table3 = [[], []]
    for i, data in enumerate(json_data_proposal):
        table3[0].append(data['time'])
        table3[1].append(data['points'])
    create_plots(map_name, table2, table3)

def main():
    path = "maptops"
    maps = [f.stem for f in pathlib.Path(path).iterdir()]
    # maps = ["bkz_cauz_short"]

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(parse, maps), total=len(maps)))

if __name__ == "__main__":
    main()
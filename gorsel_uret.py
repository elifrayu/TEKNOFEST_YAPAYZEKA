
import json
from helixai_visuals import generate_all_visuals

with open("master_results/master_report_v1.json") as f:
    report = json.load(f)

generate_all_visuals(report, output_dir="master_results")
import pandas as pd
import glob
import csv
import io

def load_data():
    path_pattern = './202*/openaq_location_783911_measurments*.csv'
    csv_files = glob.glob(path_pattern)
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file, on_bad_lines='skip')
        df_list.append(df)
    
    df_combined = pd.concat(df_list, ignore_index=True)
    df_combined['datetimeUtc'] = pd.to_datetime(df_combined['datetimeUtc'], errors='coerce').dt.tz_localize(None)
    df_combined.to_csv("data/dirty_data.csv", index=False)
    return "data/dirty_data.csv"

def trim_csv(infile, outfile):
    with open(infile, 'r', encoding='utf-8') as fin:
        content = fin.read().strip()
    
    reader = csv.reader(io.StringIO(content))
    with open(outfile, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        for row in reader:
            trimmed_row = [cell.strip() for cell in row]
            writer.writerow(trimmed_row)
    print(f"CSV trimmed and saved to {outfile}")

def clean_data(dirty_csv):
    trimmed_file = "data/trimmed_data.csv"
    trim_csv(dirty_csv, trimmed_file)
    
    df = pd.read_csv(trimmed_file)
    df_cleaned = df.drop_duplicates()
    df_cleaned.to_csv("data/cleaned_data.csv", index=False)
    return "data/cleaned_data.csv"

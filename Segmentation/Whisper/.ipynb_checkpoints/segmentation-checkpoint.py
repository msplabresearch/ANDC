#!/bin/bash 
import os, sys
from pathlib import Path
import subprocess
import json
from datetime import datetime, timedelta
from glob import glob
import pandas as pd
from tqdm import tqdm

#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
print(utils_dir)
sys.path.append(utils_dir)
from andc_args import get_args

def create_dir(dir_path):
    if os.path.exists(dir_path):
        print("Directory {} already exists".format(dir_path))
    else:
        print("Creating directory {}".format(dir_path))
        os.mkdir(dir_path)
        
def translate(json_data, filename):
    Out_Info = []
    basename = os.path.basename(filename).split('.')[0]
    for i, segment in enumerate(json_data):
        seg_text = segment['text'].strip() 
        seg_start = trim_time(timedelta(seconds=segment['start']))
        seg_end = trim_time(timedelta(seconds=segment['end']))
        out = basename+';;;'+seg_start+';;;'+seg_end+';;;'+seg_text
        Out_Info.append(out)
    return Out_Info 
        
def read_json(file_path):
    with open(file_path,'r', encoding='utf8') as rf:
        data = json.load(rf)
    return data

def trim_time(time_d):
    time_d = str(time_d)
    if '.' in time_d:
        if len(time_d.split('.')[-1])>2:
            trim_digits = len(time_d.split('.')[-1])-2
            time_d = time_d[:-trim_digits]
    else:
        time_d += '.00'
    return time_d

def get_time_delta(start_time, end_time):
    FMT = "%H:%M:%S.%f" # compute to milliseconds
    delta_time = datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)
    return delta_time.total_seconds()

def save_segments_wav(df, src_audio, output_path , audio_type = 'Short_split_file', verbose=True):
    cmd_string = "ffmpeg -hide_banner -loglevel error -i {inp} -acodec copy -ss {st} -to {en} {out}"
    if verbose:
        print("Saving {} audio files to disk".format(audio_type))
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], disable= not verbose):
        fname, start, end, text, dur = row
        out_name = os.path.join(output_path, audio_type, fname)
        command = cmd_string.format(inp=src_audio, out=out_name, st=start, en=end)
        subprocess.call(command, shell=True)
        
            
def segment_filename(row):
    return '{}_{:04d}.wav'.format(row.filename, row.name)


def json_to_df(segments, filename):
    data = []
    basename = os.path.basename(filename).split('.')[0]
    header = ['filename', 'start', 'end', 'text']
    for i, segment in enumerate(segments):
        seg_text = segment['text'].strip() 
        seg_start = trim_time(timedelta(seconds=segment['start']))
        seg_end = trim_time(timedelta(seconds=segment['end']))
        row = [basename,seg_start,seg_end,seg_text]
        data.append(row)
    return pd.DataFrame(data, columns = header)

def save_json(json_dict, root, filename):
    json_object = json.dumps(json_dict, indent=4)
    with open(os.path.join(root,filename), "w") as outfile:
        outfile.write(json_object)
        

if __name__ == "__main__":
    #directory of this file
    
    print(sys.argv)
    args = get_args()

    print("Working on :",args.root)


    root = args.root
    # path_0 = 'INPUTS_OUTPUTS'
    index_path = os.path.join(root,'INDEXER_output')
    output_path = os.path.join(root,'Outputs')

    #list of directories that we will need
    wav_path = os.path.join(index_path,'wav')
    json_path = os.path.join(index_path,'JSON_file')
    SplitTimeInfo_path = os.path.join(output_path,'JSON_parsing')
    short_output_path = os.path.join(output_path,'Short_split_file')
    long_output_path = os.path.join(output_path,'Long_split_file')


    # create output dirs
    create_dir(output_path)
    create_dir(SplitTimeInfo_path)
    create_dir(short_output_path)
    create_dir(long_output_path)
    
    
    long_dfs = []
    short_dfs = []

    for filename in glob(os.path.join(json_path,  '*.json')):
        basename = os.path.basename(filename)

        print("\nProcessing file {}".format(filename))
        segments_file = read_json(os.path.join(json_path,filename))
        df = json_to_df(segments_file, filename)

        #extend the filename by adding _1 .... _n
        df['filename'] = df.apply(segment_filename, axis=1)

        #get duration of each segment
        df['duration'] = df.apply(lambda x: get_time_delta(x.start, x.end), axis=1)

        #split segments into short [2.75s,11s) and long 11s+
        short_df = df[(df.duration >=2.75) & (df.duration < 11.0)]
        long_df = df[df.duration >= 11.0]

        short_dfs.append(short_df)
        long_dfs.append(long_df)

        #split the main audio into segements saving the .wav files (long and short segments)
        original_track = os.path.join(wav_path, basename.replace('.json', '.wav'))
        save_segments_wav(short_df, original_track, output_path, 'Short_split_file')
        save_segments_wav(long_df, original_track, output_path, 'Long_split_file')
        
    if len(long_dfs) >= 1:
        long_dfs = pd.concat(long_dfs)
        long_dfs['comment'] = 'Whisper'
        long_dfs['wav'] = [os.path.join(output_path, 'Long_split_file', x) for x in long_dfs.filename]

        sub_df = long_dfs.set_index('filename')
        sub_df.index = [x.split('.')[0] for x in sub_df.index]
        json_dict = json.loads(sub_df.to_json(orient ='index'))
        for filename, f_info in json_dict.items():
            f_info['filepaths'] = {}
            f_info['filepaths']['wav']  = f_info['wav']
            f_info.pop('wav')

        save_json(json_dict, output_path, "Long_files.json")
        
        
    if len(short_dfs) >= 1:
        short_dfs = pd.concat(short_dfs)
        short_dfs['comment'] = 'Whisper'
        short_dfs['wav'] = [os.path.join(output_path, 'Short_split_file', x) for x in short_dfs.filename]

        sub_df = short_dfs.set_index('filename')
        sub_df.index = [x.split('.')[0] for x in sub_df.index]
        json_dict = json.loads(sub_df.to_json(orient ='index'))
        for filename, f_info in json_dict.items():
            f_info['filepaths'] = {}
            f_info['filepaths']['wav']  = f_info['wav']
            f_info.pop('wav') 

        save_json(json_dict, output_path, "Short_files.json")
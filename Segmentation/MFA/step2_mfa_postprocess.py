#combines 
import os, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import textgrid
import subprocess
import shutil
from glob import glob
import shutil
from datetime import datetime, timedelta
import json

#import ANDC general arguments
file_dir = os.path.dirname(os.path.realpath(__file__))
utils_dir = os.path.join(Path(file_dir).parents[1], 'utils')
sys.path.append(utils_dir)
from andc_args import get_args


def create_dir(dir_path):
    if os.path.exists(dir_path):
        print("Directory {} already exists".format(dir_path))
    else:
        print("Creating directory {}".format(dir_path))
        os.mkdir(dir_path)
        
def parse_exist_files(root_path):
    FileNames = []
    for dirPath, dirNames, fileNames in os.walk(root_path):
        fileNames = sorted(fileNames)
        for i in range(len(fileNames)):
            fname = fileNames[i]
            fname = fname.replace('.wav','')
            fname = fname.replace('.txt','')
            fname = fname.replace('.TextGrid','')
            FileNames.append(fname)
    FileNames = np.unique(np.array(FileNames)).tolist()  
    return FileNames

def save_segment_text(out_name, text):
    with open(out_name,'w', encoding='utf8') as wf:
        wf.write(text)
        wf.close()
        
        
def read_segment_text(file_name):
    with open(file_name) as f:
        lines = f.read()
    return lines


def trim_time(time_d):
    time_d = str(time_d)
    if '.' in time_d:
        if len(time_d.split('.')[-1])>2:
            trim_digits = len(time_d.split('.')[-1])-2
            time_d = time_d[:-trim_digits]
    else:
        time_d += '.00'
    return time_d


##############################################################################

    
if __name__ == "__main__": 
    
    
    # Parameters
    args = get_args()
    root = args.root
    output_root = os.path.join(root, 'Outputs')


    output_short_file = os.path.abspath(os.path.join(output_root, 'Short_split_file'))


    reseg_output_file = os.path.join(output_root,'Long_reseg_file')
    mfa_corpus_root = os.path.join(output_root,'mfa_corpus/speaker1')
    mfa_aligned_root = os.path.join(output_root,'mfa_align_rsl/speaker1')
    seg_threshold = 0.3 # in secs


    with open(os.path.join(output_root, "Short_files.json"), "r") as openfile:
        short_files = json.load(openfile)

    with open(os.path.join(output_root, "Long_files.json"), "r") as openfile:
        long_files = json.load(openfile)

    create_dir(reseg_output_file)


    # get info for any non-align files
    missing_files = list(set(parse_exist_files(mfa_corpus_root))-set(parse_exist_files(mfa_aligned_root)))
    missing_files = sorted(missing_files)
    print("*****NOTE: Missing Aligned Files= "+str(len(missing_files))+" *****")





    # main function
    aligned_files_all = sorted(os.listdir(mfa_aligned_root))
    for filepath in glob(os.path.join(mfa_aligned_root,'*.TextGrid')):

        filename = os.path.basename(filepath).split('.')[0]
        align_rsl = textgrid.TextGrid.fromFile(filepath)[0]


        # obtain split timestamps
        seg_times = []
        seg_asr = ["",]
        for j in range(len(align_rsl)):
            word = align_rsl[j].mark
            word_t_start = align_rsl[j].minTime
            word_t_end = align_rsl[j].maxTime
            duration = word_t_end - word_t_start
            if word=="" and duration>=seg_threshold:
                seg_times.append(str((word_t_start+word_t_end)/2))
                seg_asr.append("")
            elif word == "":
                continue
            else:
                seg_asr[-1] += word + ' ' 

        #removing pre/post spaces and remove last empty element
        seg_asr = [x.strip() for x in seg_asr]

        assert len(seg_asr)-1 == len(seg_times)

        #update meta-data to reflect MFA data (optional)
        long_files[filename]['mfa']  = {}
        long_files[filename]['mfa']['splits'] = seg_times
        long_files[filename]['mfa']['texts'] = seg_asr 

        # run the ffmpeg cmd
        cmd_string = 'ffmpeg -hide_banner -loglevel error -i {input} -f segment -segment_times {times} -c copy -map 0 {output}'
        input_file = os.path.join(mfa_corpus_root, filename+'.wav')
        output_file = os.path.join(reseg_output_file , filename+'_%04d.wav')
        splits = ",".join(seg_times)
        command = cmd_string.format(input=input_file, output=output_file, times=splits)
        subprocess.call(command, shell=True)



    # remove files that are not within 2.75~11 secs
    split_files_all = sorted(glob(reseg_output_file+'/*'))



    for audio_clip in split_files_all:
        basename = os.path.basename(audio_clip)
        short_name = basename.split('.')[0] #without extendion
        seg_num = int(short_name[-4:]) #get last 4digits
        original_clip = short_name[:-5] #remove last 4 digit to get original name
        original_meta = long_files[original_clip]

        sr, data = wavfile.read(audio_clip)
        duration = len(data)/float(sr)

        asr_file = audio_clip.replace('Long_reseg_file', 'Long_reseg_ASR')
        asr_file = asr_file.replace('.wav', '.txt')
        asr_text = original_meta['mfa']['texts'][seg_num]
        segments = len(original_meta['mfa']['splits'])

        #get start time of current segment
        if seg_num == 0:   
            start_time = original_meta['start']
        else:
            start_time = trim_time(timedelta(seconds=float(original_meta['mfa']['splits'][seg_num-1])))

        #get end time of current segmenet
        if seg_num == segments:
            end_time = original_meta['end']
        else:
            end_time = trim_time(timedelta(seconds=float(original_meta['mfa']['splits'][seg_num])))

        if len(asr_text.split()) > 3:
            low_word_ct = False
        else:
            low_word_ct = True
        # #remove audio/asr if segment is too long/short or if it has under 3 words
        if duration<2.75 or duration>11.1 or low_word_ct:
            continue
        #else add it to the list of short clips
        else:
            shutil.move(audio_clip, output_short_file)
            short_files[short_name] = {}
            short_files[short_name]['start'] = start_time
            short_files[short_name]['end'] = end_time
            short_files[short_name]['duration'] = duration
            short_files[short_name]['comment'] = 'MFA'
            short_files[short_name]['text'] = asr_text
            short_files[short_name]['filepaths'] = {}
            short_files[short_name]['filepaths']['wav'] = os.path.join(output_short_file, basename)

    #update the json files based on the resegmentation
    json_object = json.dumps(short_files, indent=4)
    with open(os.path.join(output_root, "Short_files.json"), "w") as outfile:
        outfile.write(json_object)

    json_object = json.dumps(long_files, indent=4)
    with open(os.path.join(output_root, "Long_files.json"), "w") as outfile:
        outfile.write(json_object)


    #remove temporary directory used to run and filter mfa output
    # shutil.rmtree(reseg_output_file)
    # shutil.rmtree(os.path.split(mfa_corpus_root)[0])
    # shutil.rmtree(os.path.split(mfa_aligned_root)[0])
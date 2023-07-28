import os
import librosa

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

class Wav2VecExtractor:
    def __init__(self):
        self.preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to('cuda')
        self.model.eval()

    def extract_all(self, raw_wav_list):
        wav2vec_list = []
        for raw_wav in tqdm(raw_wav_list):
            wav_input_16khz = torch.Tensor(raw_wav)
            wav_input_16khz = wav_input_16khz.cuda()
            x = self.preprocessor(wav_input_16khz, return_tensors="pt", sampling_rate=16000, padding="longest").input_values
            x = x.cuda()
            with torch.no_grad():
                z = self.model(x).last_hidden_state
            z = z.squeeze(0).cpu().numpy()
            wav2vec_list.append(z)
        return wav2vec_list


class MelSpecExtractor:
    def __init__(self, *args, **kwargs):
        self.sr = kwargs.get("sampling_rate", 16000)
        self.n_fft = kwargs.get("window_size", 400) # 16 * 25ms
        self.hop_length = kwargs.get("step_size", 160) # 16 * 10ms
        return

    def read_and_extract_all(self, wav_path_list, get_raw_wav=False):
        mel_spec_list = []
        if get_raw_wav:
            raw_wav_list = []
        for wav_path in tqdm(wav_path_list):
            wav_input, _ = librosa.load(wav_path, sr=self.sr)
            mel_spec = librosa.feature.melspectrogram(y=wav_input, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_spec_list.append(mel_spec)
            if get_raw_wav:
                raw_wav_list.append(wav_input)
        result = mel_spec_list
        if get_raw_wav:
            result = (result, raw_wav_list)
        return result

def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav


class WavExtractor:
    def __init__(self, *args, **kwargs):
        self.wav_path_list = kwargs.get("wav_paths", args[0])
        # self.sr = kwargs.get("sampling_rate", 16000)
        self.nj = kwargs.get("nj", 24)
    def extract(self):
        print("Extracting wav files")
        with Pool(self.nj) as p:
            wav_list = list(tqdm(p.imap(extract_wav, self.wav_path_list), total=len(self.wav_path_list)))
    
        return wav_list

def DynamicChunkForAll(All_data, m, C, n):
    def DynamicChunk(Batch_data, m, C, n):
        """
        Note! This function can't process sequence length which less than given m=62
        (e.g., 1sec=62frames, if LLDs extracted by hop size 16ms then 16ms*62=0.992sec~=1sec)
        Please make sure all your input data's length are greater then given m.
        
        Args:
            Batch_data$ (list): list of data arrays for a single batch.
            Batch_label$ (list): list of training targets for a single batch.
                    m$ (int) : chunk window length (i.e., number of frames within a chunk)
                    C$ (int) : number of chunks splitted for a sentence
                    n$ (int) : scaling factor to increase number of chunks splitted in a sentence
        """
        num_shifts = n*C-1  # Tmax = 11sec (for the MSP-Podcast corpus), 
                            # chunk needs to shift 10 times to obtain total C=11 chunks for each sentence
        Split_Data = []
        data = Batch_data
        Duration = data.size(0)
        # window-shifting size varied by differenct length of input utterance => dynamic step size
        step_size = int(int(Duration-m)/num_shifts)      
        # Calculate index of chunks
        start_idx = [0]
        end_idx = [m]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # Output Split Data
        for iii in range(len(start_idx)):
            cur_chunk = data[start_idx[iii]: end_idx[iii]]
            cur_chunk = cur_chunk.transpose(1,0)
            Split_Data.append(cur_chunk.unsqueeze(0))
        result = torch.cat(Split_Data, dim=0)
        return result

    result = []
    for data in All_data:
        chunk_data = DynamicChunk(data, m, C, n)
        result.append(chunk_data.unsqueeze(0))
    result = torch.cat(result, dim=0)
    return result


def unpack_torch_segment(padded_segment, duration):
    batch_num = padded_segment.size(0)
    result = []
    for idx in range(batch_num):
        cur_segment = padded_segment[idx]
        
        cur_dur = duration[idx]
        cut_seg = cur_segment[:cur_dur]
        result.append(cut_seg)
    resutl = torch.Tensor(result)
    return result

def AverageAll(All_data):
    # print(All_data.size())
    # for batch_idx, cur_dur in enumerate(duration):
    #     cur_batch = All_data[batch_idx][:cur_dur]
    #     print(cur_batch.size())
    All_data = torch.mean(All_data, dim=1)

    return All_data
   

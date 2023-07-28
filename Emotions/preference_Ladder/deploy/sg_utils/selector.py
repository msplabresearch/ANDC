import librosa
import numpy as np
import soundfile as sf

class Selector:
    def __init__(self):
        self.duration_map={
            "1m": 16000*60*1,
            "5m": 16000*60*5,
            "10m": 16000*60*10,
            "30m": 16000*60*30,
            "1h": 16000*60*60*1,
            "3h": 16000*60*60*3,
            "7h": 16000*60*60*7,
            "15h": 16000*60*60*15,
            "30h": 16000*60*60*30,
            "full": None
        }
    
    def select(self, wav_list, max_dur, criterion="random"):
        max_dur_key = max_dur
        max_dur = self.duration_map[max_dur]
        if max_dur == None:
            return wav_list

        if criterion == "random":
            shuffled_list = np.random.permutation(wav_list)
            result_list = []
            cur_dur = 0
            for wav_path in wav_list:
                data, sr = librosa.load(wav_path, sr=16000)
                assert sr == 16000
                cur_dur += len(data)
                if cur_dur > max_dur:
                    if len(result_list) == 0:
                        sidx = np.random.randint(0, len(data)-max_dur)
                        data = data[sidx:sidx+max_dur]
                        new_wav_path = "temp_"+str(max_dur_key)+".wav"
                        sf.write(new_wav_path, data, 16000)
                        result_list.append(new_wav_path)
                    break
                result_list.append(wav_path)
            return result_list
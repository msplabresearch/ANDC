import librosa
import numpy as np
import soundfile
import torch

def add_gain(clean_wav, gain=2):
    noisy_wav = clean_wav * gain
    return noisy_wav

def add_convolution(clean_wav, rir_wav):
    clean_dur = len(clean_wav)
    noisy_wav = np.convolve(clean_wav, rir_wav)[:clean_dur]
    return noisy_wav

def calc_noise_gain(clean_wav, noise_wav, SNR):
    clean_power = np.mean(np.abs(clean_wav))
    noise_power = np.mean(np.abs(noise_wav))
    noise_gain = clean_power / (noise_power * (10 ** (SNR/10)) )
    return noise_gain

def add_additive(clean_wav, noise_wav, SNR):
    noise_gain = calc_noise_gain(clean_wav, noise_wav, SNR)
    noisy_wav = clean_wav + noise_gain * noise_wav
    return noisy_wav

def add_white_noise(clean_wav, SNR=10):
    clean_dur = len(clean_wav)
    white_noise = np.random.randn(clean_dur)
    noisy_wav = add_additive(clean_wav, white_noise)
    return noisy_wav
    
def contaminate_all(clean_wavs, noise_wavs, SNR, is_torch=False):
    is_mct = True if type(SNR) == list else False
    is_multiple_noise = True if len(noise_wavs) > 1 else False
    result_noisy_wavs = []
    

    
    for clean_wav in clean_wavs:
        clean_dur = len(clean_wav)  
        if is_multiple_noise:
            noise_sample = noise_wavs[np.random.randint(0, len(noise_wavs))]
            noise_dur = len(noise_sample)
            if clean_dur > noise_dur:
                cur_noise_wav = np.zeros(clean_dur)
                cur_noise_wav[:noise_dur] = noise_sample
            else:
                cur_noise_wav = noise_sample
            noise_dur = len(cur_noise_wav)
            if noise_dur-clean_dur != 0:
                noise_sidx = np.random.randint(0, noise_dur-clean_dur)  
            else:
                noise_sidx = 0
            cur_noise_wav = np.array(cur_noise_wav[noise_sidx:noise_sidx+clean_dur])
        if not is_multiple_noise:
            noise_dur = len(noise_wavs[0])
            noise_sidx = np.random.randint(0, noise_dur-clean_dur)  
            cur_noise_wav = noise_wavs[0][noise_sidx:noise_sidx+clean_dur]
        cur_snr = SNR[np.random.randint(0, len(SNR))] if is_mct else SNR
        cur_noisy_wav = add_additive(clean_wav, cur_noise_wav, cur_snr)
        result_noisy_wavs.append(cur_noisy_wav)
    if is_torch:
        result_noisy_wavs = torch.Tensor(result_noisy_wavs).cuda(non_blocking=True).float()
    return result_noisy_wavs
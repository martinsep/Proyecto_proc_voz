import argparse
import torch
from torch import nn
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
import numpy as np
import IPython
import soundfile as sf
import glob
from pesq import pesq
from pystoi import stoi
from os import path
from tqdm import tqdm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--referencia", '-referencia',help="carpeta audios de referencia", default = "./AudiosProcesados_Cortos/Clean_ch1_Static1", type = str)
    parser.add_argument("--folder", '-folder',help="carpeta audios", default = "./AudiosProcesados_Cortos/Benforming_DS_angle_Static1", type = str)
    parser.add_argument("--result_csv", '-result_csv',help="Archivo csv para escribir las metricas", default = "./Resultados.csv", type = str)
    parser.add_argument("--vad", '-vad',help="path vad", default = "", type = str)
    return parser.parse_args()

    
    
def snr_vad(wav,ref,fs,vad):    
    model, utils = vad    
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils    
    speech_timestamps = get_speech_timestamps(ref, model, sampling_rate=fs)

    speech_mask=np.zeros(ref.shape)
    for stamps in speech_timestamps:
        speech_mask[stamps["start"]:stamps["end"]]=1;
    noise_mask=1-speech_mask
    power_noise=np.einsum('i,i,i',wav,wav,noise_mask)/np.sum(noise_mask)
    power_speech=np.einsum('i,i,i',wav,wav,speech_mask)/np.sum(speech_mask)
    speech_content=np.sum(speech_mask)/(np.sum(speech_mask)+np.sum(noise_mask))
    return 10*np.log10(power_speech/power_noise) , 10*np.log10(speech_content*power_speech/power_noise)
    
def mos2rawpesq(x):
   a= 0.999
   b= 4.999
   c= -1.4945
   d= 4.6607
   return (np.log(((b-x)/(x-a)))-d)/c    
   
args = parse_args()
vad = torch.hub.load(source='local',repo_or_dir=args.vad,model='silero_vad',force_reload=False)


D=glob.glob(args.referencia + '/*.wav')

N_signals=0
Total_SNR=0
Total_SNR_compensado=0
Total_SI_SNR=0
Total_PESQ=0
Total_STOI=0
for f in tqdm(D):
    name =f.split("/")[-1].split(".")[0]
    if path.isfile(args.folder + '/' + f.split("/")[-1]):
        #Carga los audios y segmenta al mismo largo
        ref, samplerate = sf.read(f)
        #print(ref.shape)
        wav, samplerate = sf.read(args.folder + '/' + f.split("/")[-1])
        length=np.minimum(ref.shape[0],wav.shape[0])
        ref=ref[0:length]
        wav=wav[0:length]

        SNR,SNR_compensado=snr_vad(wav,ref,samplerate,vad)
        #print(SNR)
        SI_SNR=scale_invariant_signal_noise_ratio(torch.tensor(wav),torch.tensor(ref)).numpy()
        PESQ=mos2rawpesq(pesq(samplerate, ref, wav, 'nb'))
        STOI=stoi(ref,wav,samplerate)
        
        N_signals=N_signals+1
        Total_SNR=Total_SNR+SNR
        Total_SNR_compensado=Total_SNR_compensado+SNR_compensado
        Total_SI_SNR=Total_SI_SNR+SI_SNR
        Total_PESQ=Total_PESQ+PESQ
        Total_STOI=Total_STOI+STOI

Total_SNR=Total_SNR/N_signals
Total_SNR_compensado=Total_SNR_compensado/N_signals
Total_SI_SNR=Total_SI_SNR/N_signals
Total_PESQ=Total_PESQ/N_signals
Total_STOI=Total_STOI/N_signals


path = f'{args.result_csv}'
try:
   df = pd.read_csv(path, header=0)
except:
   columns = ['Experimento', 'N_Utt','SNR', 'PESQ', 'STOI']
   df = pd.DataFrame(columns=columns)

data = pd.Series({'Experimento': args.folder.split("/")[-1],
                    'N_Utt': N_signals, 
                    'SNR': Total_SNR, 
                    'PESQ': Total_PESQ, 
                    'STOI': Total_STOI})

df = pd.concat([df, data.to_frame().T], ignore_index=True)
df.to_csv(path, index=False)
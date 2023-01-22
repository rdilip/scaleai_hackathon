import streamlit as st
import torch
print(torch.__version__)

from midiSynth.synth import MidiSynth
midi_synth = MidiSynth()

import time
import os
from copy import deepcopy

import requests
import pickle

import sys
sys.path.append("/home/ubuntu/EMOPIA/workspace/transformer")
from utils import write_midi
from models import TransformerModel, network_paras
import json
from glob import glob


def spellbook_api(inp):
    data = {
      "input": inp
    }
    headers = {"Authorization": "Basic cld6eag39004csr1ao7f7po5s"}
    response = requests.post(
      "https://dashboard.scale.com/spellbook/api/app/1f1763jry",
      json=data,
      headers=headers
    )
    return response

def generate_music(inp, path_outfile: str="test", log: dict=None):
    response = spellbook_api(inp)
    if not response:
        print("Spellbook is not responsive!")
    hv = "high-valence"
    lv = "low-valence"
    la = "low-arousal"
    ha = "high-arousal"
    emotion_tag = 1
    response_text = json.loads(response.text)["text"]
    if hv in response_text and ha in response_text:
        emotion_tag = 1
    elif ha in response_text and lv in response_text:
        emotion_tag = 2
    elif lv in response_text and la in response_text:
        emotion_tag = 3
    else:
        emotion_tag = 4
    print("="*20)
    print(f"EMOTION TAG: {emotion_tag}")
    print("="*20)
    res, _ = net.inference_from_scratch(dictionary, emotion_tag, n_token=8, display=False)
    write_midi(res, path_outfile + '.mid', word2event)
    midi_synth.play_midi(path_outfile + '.mid')
    midi_synth.midi2audio(path_outfile + '.mid', path_outfile + '.mp3')

    os.system(f"ffmpeg -i {path_outfile}.mp3 -ar 48000 -vn -c:a libvorbis {path_outfile}.ogg")
    log[inp] = path_outfile
    with open(f"audio_files/log.json", "w+") as f:
        json.dump(log, f)

    return res, log

# Prepare dictionary
path_dictionary = "/home/ubuntu/EMOPIA/dataset/co-representation/dictionary.pkl"
assert os.path.exists(path_dictionary)

log = {}

with open(path_dictionary, "rb") as f:
    dictionary = pickle.load(f)
event2word, word2event = dictionary

n_class = []
for key in event2word.keys():
    n_class.append(len(dictionary[0][key]))
n_token = len(n_class)

path_to_ckpt = "/home/ubuntu/EMOPIA/exp/pretrained_transformer/loss_25_params.pt"
assert os.path.exists(path_to_ckpt)

net = TransformerModel(n_class, is_training=False)
net.cuda()
net.eval()

net.load_state_dict(torch.load(path_to_ckpt))

st.title("Cognituner")

script = st.text_area("Enter your script here!", value="")
path_outfile=f"audio_files/{time.time()}"
music, log = generate_music(script, path_outfile=path_outfile, log=log)
log[str(script)] = path_outfile


audio_file = open(f"{path_outfile}.ogg", 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg')

st.header("Examples")

for fname in glob("sample_files/Q*.ogg"):
    descr = os.path.splitext(fname)[0].split("/")[1].replace("_", " ")[3:]
    st.subheader(descr)
    audio_file = open(fname, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

for fname in glob("script_files/*.ogg"):
    base = os.path.splitext(fname)[0].split("/")[1]
    print(base, fname)
    with open(f"script_files/{base}.txt", 'r') as file:
        script = file.read().replace('\n', '')

    descr = base.replace("_", " ")
    st.subheader(descr)
    st.markdown(script)
    audio_file = open(fname, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')
# full_replacement_chart = st.empty()
# audio_canvas = st.empty()



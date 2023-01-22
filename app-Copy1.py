import streamlit as st
import torch
print(torch.__version__)

from midiSynth.synth import MidiSynth
midi_synth = MidiSynth()

import os

import requests
import pickle

import sys
sys.path.append("/home/ubuntu/EMOPIA/workspace/transformer")
from utils import write_midi
from models import TransformerModel, network_paras
import json

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

def generate_music(inp, path_outfile: str="test"):
    response = spellbook_api(inp)
    if not response:
        display(HTML("<h1>Try again</h1>"))
    hv = "high-valence"
    lv = "low-valence"
    la = "low-arousal"
    ha = "high-arousal"
    emotion_tag = 1
    response_text = json.loads(response.text)["text"]
    if hv and ha in response_text:
        emotion_tag = 1
    elif ha and lv in response_text:
        emotion_tag = 2
    elif lv and la in response_text:
        emotion_tag = 3
    else:
        emotion_tag = 4
    res, _ = net.inference_from_scratch(dictionary, emotion_tag, n_token=8, display=False)
    write_midi(res, path_outfile + '.mid', word2event)
    midi_synth.play_midi(path_outfile + '.mid')
    midi_synth.midi2audio(path_outfile + '.mid', path_outfile + '.mp3')

    os.system(f"ffmpeg -i {path_outfile}.mp3 -ar 48000 -vn -c:a libvorbis {path_outfile}.ogg")

    return res

# Prepare dictionary
path_dictionary = "/home/ubuntu/EMOPIA/dataset/co-representation/dictionary.pkl"
assert os.path.exists(path_dictionary)

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

st.write("Script2Muse")
script = st.text_area("Enter your script here!", value="")

# Input code to go from a script to a code using spellbook
music = generate_music(script)

audio_file = open('test.ogg', 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format='audio/ogg')
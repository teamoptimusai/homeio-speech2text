from glob import glob
import json
import os
import random

with open('dataset.json', 'w') as file:
    for label_filepath in glob("LibriSpeech/train-clean-100/*/*/*.txt"):
        dir = "/".join(label_filepath.split('/')[:-1])
        save_dir = 'LibriSpeech_Dataset'
        os.makedirs(save_dir, exist_ok=True)
        with open(label_filepath, 'r') as f:
            lines = f.readlines()
        lines = [(line.split()[0], " ".join(line.split()[1:]))
                 for line in lines]
        for audio_file, text in lines:
            command = "ffmpeg -i {}/{}.flac {}/{}.wav".format(
                dir, audio_file, save_dir, audio_file)
            os.system(command)
            file.write(json.dumps(
                {'key': f'datasets/{save_dir}/{audio_file}.wav', 'text': text})+"\n")

train_percentage = 0.8
with open('dataset.json', 'r') as file:
    lines = file.readlines()
    random.shuffle(lines)
    train_lines = lines[:int(len(lines)*train_percentage)]
    test_lines = lines[int(len(lines)*train_percentage):]
with open(f'{save_dir}/train.json', 'w') as file:
    file.writelines(train_lines)
with open(f'{save_dir}/valid.json', 'w') as file:
    file.writelines(test_lines)

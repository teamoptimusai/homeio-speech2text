import json
import random
import csv
from pydub import AudioSegment
from tqdm import tqdm
import os


def convert(row):
    in_audio_path = row['path']
    out_audio_path = in_audio_path.rpartition('.')[0] + ".wav"
    text = row['sentence']
    dataset.append(
        {"key": f'datasets/{audio_outdir}/{out_audio_path}', "text": text})
    sound = AudioSegment.from_mp3(f'{clips_dir}/{in_audio_path}')
    sound.export(f'{audio_outdir}/{out_audio_path}', format="wav")


def main():
    with open(tsv_file, newline='') as tsv:
        reader = csv.DictReader(tsv, delimiter='\t')
        # for row in tqdm(reader):
        #     convert(row)
        try:
            from multiprocessing import Pool
            pool = Pool(os.cpu_count())
            pool.map(convert, reader)
        except ImportError:
            for row in tqdm(reader):
                convert(row)
        finally:
            pool.close()
            pool.join()

    random.shuffle(dataset)
    num_data = len(dataset)
    print(f'num_data: {num_data}')
    with open(f'train.json', 'w') as f:
        for r in tqdm(dataset[:int(num_data*split_percent)], desc='Creating train.json'):
            line = json.dumps(r)
            f.write(line + "\n")
    with open(f'valid.json', 'w') as f:
        for r in tqdm(dataset[int(num_data*split_percent):], desc='Creating valid.json'):
            line = json.dumps(r)
            f.write(line + "\n")


if __name__ == "__main__":
    tsv_file = "cv-corpus-8.0-2022-01-19/en/validated.tsv"
    clips_dir = "cv-corpus-8.0-2022-01-19/en/clips"
    audio_outdir = "cv-corpus-8.0-2022-01-19-wav"
    os.makedirs(audio_outdir)
    split_percent = 0.8

    dataset = []

    main()

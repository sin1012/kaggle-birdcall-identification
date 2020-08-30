# https://www.kaggle.com/c/birdsong-recognition/discussion/159943
# https://www.kaggle.com/c/birdsong-recognition/discussion/164197
# https://github.com/koukyo1994/kaggle-birdcall-resnet-baseline-training/blob/master/input/birdsong-recognition/prepare.py

import argparse
import soundfile as sf
import warnings
from tqdm import tqdm
import librosa
import pandas as pd
from pathlib import Path
from joblib import delayed, Parallel


def resample(df: pd.DataFrame, target_sr: int):
    audio_dir = Path("./birdsong-recognition/train_audio")
    resample_dir = Path("./birdsong-recognition/train_audio_resampled")
    resample_dir.mkdir(exist_ok=True, parents=True)
    warnings.simplefilter("ignore")

    for i, row in tqdm(df.iterrows()):
        ebird_code = row.ebird_code
        filename = row.filename
        ebird_dir = resample_dir / ebird_code
        if not ebird_dir.exists():
            ebird_dir.mkdir(exist_ok=True, parents=True)

        try:
            y, _ = librosa.load(
                str(audio_dir / ebird_code / filename),
                sr=target_sr, mono=True, res_type="kaiser_fast")
            #print(len(y))

            filename = filename.replace(".mp3", ".wav")
            sf.write(str(ebird_dir / filename), y, samplerate=target_sr)
        except Exception:
            with open("skipped.txt", "a") as f:
                file_path = str(audio_dir / ebird_code / filename)
                f.write(file_path + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", default=32000, type=int)
    parser.add_argument("--n_splits", default=8, type=int)
    args = parser.parse_args()

    target_sr = args.sr

    train = pd.read_csv("./birdsong-recognition/train.csv")
    dfs = []
    for i in range(args.n_splits):
        if i == args.n_splits - 1:
            start = i * (len(train) // args.n_splits)
            df = train.iloc[start:, :].reset_index(drop=True)
            dfs.append(df)
        else:
            start = i * (len(train) // args.n_splits)
            end = (i + 1) * (len(train) // args.n_splits)
            df = train.iloc[start:end, :].reset_index(drop=True)
            dfs.append(df)

    Parallel(
        n_jobs=args.n_splits,
        verbose=10)(delayed(resample)(df, args.sr) for df in dfs)

    train["resampled_sampling_rate"] = target_sr
    train["resampled_filename"] = train["filename"].map(
        lambda x: x.replace(".mp3", ".wav"))
    train["resampled_channels"] = "1 (mono)"
    train.to_csv("train_mod.csv", index=False)
import multiprocessing

import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.audio.preparation.spleeter_utils.filter_noisy_clips_collector import invert_spectrogram_and_save
from scripts.audio.preparation.spleeter_utils.spleeter_dataset import SpleeterDataset
from scripts.audio.preparation.spleeter_utils.spleeter_separator_mod import Separator




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--out')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--partition_size', default=None)
    parser.add_argument('--partition', default=None)
    args = parser.parse_args()

    src_dir = args.path
    output_sample_rate=22050
    resume_file = args.resume

    worker_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(target=invert_spectrogram_and_save, args=(args, worker_queue))
    worker.start()

    loader = DataLoader(SpleeterDataset(src_dir, batch_sz=16, sample_rate=output_sample_rate,
                                        max_duration=10, partition=args.partition, partition_size=args.partition_size,
                                        resume=resume_file), batch_size=1, num_workers=1)

    separator = Separator('spleeter:2stems', multiprocess=False)
    for batch in tqdm(loader):
        audio, files, ends, stft = batch['audio'], batch['files'], batch['ends'], batch['stft']
        sep = separator.separate_spectrogram(stft.squeeze(0).numpy())
        worker_queue.put((sep['vocals'], sep['accompaniment'], audio.shape[1], files, ends))
    worker_queue.put(None)
    worker.join()


if __name__ == '__main__':
    main()

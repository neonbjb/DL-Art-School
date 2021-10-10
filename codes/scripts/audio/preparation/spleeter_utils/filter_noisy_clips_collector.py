from scripts.audio.preparation.spleeter_utils.spleeter_separator_mod import Separator
import numpy as np

def invert_spectrogram_and_save(args, queue):
    separator = Separator('spleeter:2stems', multiprocess=False, load_tf=False)
    out_file = args.out
    unacceptable_files = open(out_file, 'a')

    while True:
        combo = queue.get()
        if combo is None:
            break
        vocals, bg, wavlen, files, ends = combo
        vocals = separator.stft(vocals, inverse=True, length=wavlen)
        bg = separator.stft(bg, inverse=True, length=wavlen)
        start = 0
        for path, end in zip(files, ends):
            vmax = np.abs(vocals[start:end]).mean()
            bmax = np.abs(bg[start:end]).mean()
            start = end

            # Only output to the "good" sample dir if the ratio of background noise to vocal noise is high enough.
            ratio = vmax / (bmax+.0000001)
            if ratio < 18:  # These values were derived empirically
                unacceptable_files.write(f'{path[0]}\n')
        unacceptable_files.flush()

    unacceptable_files.close()
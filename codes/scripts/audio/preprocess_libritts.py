# Combines all libriTTS WAV->text mappings into a single file
import os

from tqdm import tqdm

if __name__ == '__main__':
    libri_root = 'E:\\audio\\LibriTTS'
    basis = 'train-other-500'

    readers = os.listdir(os.path.join(libri_root, basis))
    ofile = open(os.path.join(libri_root, f'{basis}_list.txt'), 'w', encoding='utf-8')
    for reader_dir in tqdm(readers):
        reader = os.path.join(libri_root, basis, reader_dir)
        if not os.path.isdir(reader):
            continue
        for chapter_dir in os.listdir(reader):
            chapter = os.path.join(reader, chapter_dir)
            if not os.path.isdir(chapter):
                continue
            id = f'{os.path.basename(reader)}_{os.path.basename(chapter)}'
            trans_file = f'{id}.trans.tsv'
            with open(os.path.join(chapter, trans_file), encoding='utf-8') as f:
                trans_lines = [line.strip().split('\t') for line in f]
                for line in trans_lines:
                    wav_file, raw_text, normalized_text = line
                    wav_file = '/'.join([basis, reader_dir, chapter_dir, f'{wav_file}.wav'])
                    if not os.path.exists(os.path.join(libri_root, wav_file)):
                        print(f'!WARNING could not open {wav_file}')
                    ofile.write(f'{wav_file}|{normalized_text}\n')
            ofile.flush()
    ofile.close()

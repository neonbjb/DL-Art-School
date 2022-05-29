import os

if __name__ == '__main__':
    basepath = 'Y:\\bigasr_dataset\\hifi_tts'

    english_file = os.path.join(basepath, 'transcribed-oco-realtext.tsv')
    if not os.path.exists(english_file):
        english_file = os.path.join(basepath, 'transcribed-oco.tsv')
    phoneme_file = os.path.join(basepath, 'transcribed-phoneme-oco.tsv')

    texts = {}
    with open(english_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            spl = line.split('\t')
            if len(spl) == 3:
                text, p, _ = spl
                texts[p] = text
            else:
                print(f'Error processing line {line}')

    with open(phoneme_file, 'r', encoding='utf-8') as f:
        wf = open(os.path.join(basepath, 'transcribed-phoneme-english-oco.tsv'), 'w', encoding='utf-8')
        for line in f.readlines():
            spl = line.split('\t')
            if len(spl) == 3:
                _, p, codes = spl
                codes = codes.strip()
                if p not in texts:
                    print(f'Could not find the text for {p}')
                    continue
                wf.write(f'{texts[p]}\t{p}\t{codes}\n')
        wf.close()

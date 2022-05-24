from itertools import groupby

import torch
from transformers import Wav2Vec2CTCTokenizer

from models.audio.tts.ctc_code_generator import CtcCodeGenerator


def get_ctc_metadata(codes):
    if isinstance(codes, torch.Tensor):
        codes = codes.tolist()
    grouped = groupby(codes)
    rcodes, repeats, pads = [], [], [0]
    for val, group in grouped:
        if val == 0:
            pads[-1] = len(list(
                group))  # This is a very important distinction! It means the padding belongs to the character proceeding it.
        else:
            rcodes.append(val)
            repeats.append(len(list(group)))
            pads.append(0)

    rcodes = torch.tensor(rcodes)
    # These clip values are sane maximum values which I did not see in the datasets I have access to.
    repeats = torch.clip(torch.tensor(repeats), min=1, max=30)
    pads = torch.clip(torch.tensor(pads[:-1]), max=120)

    return rcodes, pads, repeats


if __name__ == '__main__':
    model = CtcCodeGenerator(model_dim=512, layers=16, dropout=0).eval().cuda()
    model.load_state_dict(torch.load('../experiments/train_encoder_build_ctc_alignments_toy/models/76000_generator_ema.pth'))

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('jbetker/tacotron-symbols')
    text = "and now, what do you want."
    seq = [0, 0, 0, 38, 51, 51, 41, 11, 11, 51, 51, 0, 0, 0, 0, 52, 0, 60, 0, 0, 0, 0, 0, 0, 6, 11, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 45, 0, 38, 57, 57, 11, 0, 41, 52, 52, 11, 11, 62, 52, 52, 58, 0, 11, 11, 60, 0, 0, 0, 0, 38, 0, 0, 51, 51, 0, 0, 57, 0, 0, 7, 7, 0, 0, 0]
    codes, pads, repeats = get_ctc_metadata(seq)

    with torch.no_grad():
        codes = codes.cuda().unsqueeze(0)
        pads = pads.cuda().unsqueeze(0)
        repeats = repeats.cuda().unsqueeze(0)

        ppads = pads.clone()
        prepeats = repeats.clone()
        mask = torch.zeros_like(pads)
        conf_str = tokenizer.decode(codes[0].tolist())
        for s in range(codes.shape[-1]):
            logits, confidences = model.inference(codes, pads * mask, repeats * mask)

            confidences = confidences * mask.logical_not()  # prevent prediction of tokens that have already been predicted.
            i = confidences.argmax(dim=-1)
            pred = logits[0,i].argmax()

            pred_pads = pred % model.max_pad
            pred_repeats = pred // model.max_pad
            ppads[0,i] = pred_pads
            prepeats[0,i] = pred_repeats
            mask[0,i] = 1

            conf_str = conf_str[:i] + conf_str[i].upper() + conf_str[i+1:]
            print(f"conf: {conf_str} pads={pred_pads}:{pads[0,i].item()} repeats={pred_repeats}:{repeats[0,i].item()}")
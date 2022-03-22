import torch

from utils.util import load_model_from_config

if __name__ == '__main__':
    config = "D:\\dlas\\options\\train_wav2vec_matcher.yml"
    model_name = "generator"
    model_path = "D:\dlas\experiments\train_wav2vec_matcher\models"
    wav_dump_path = "FIXME"

    model = load_model_from_config(config, model_name, also_load_savepoint=False, load_path=model_path, device='cuda').eval()
    w2v_logits, audio_samples = torch.load(wav_dump_path)

    w2v_logits_chunked = torch.chunk(w2v_logits, 32)
    for chunk in w2v_logits_chunked:

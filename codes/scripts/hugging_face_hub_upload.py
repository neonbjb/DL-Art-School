if __name__ == '__main__':
    """
    Utility script for uploading model weights to the HF hub
    """

    '''
    model = Wav2VecWrapper(vocab_size=148, basis_model='facebook/wav2vec2-large-robust-ft-libri-960h', freeze_transformer=True, checkpointing_enabled=False)
    weights = torch.load('D:\\dlas\\experiments\\train_wav2vec_mass_large2\\models\\22500_wav2vec.pth')
    model.load_state_dict(weights)
    model.w2v.save_pretrained("jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli", push_to_hub=True)
    '''

    # Build tokenizer vocab
    #mapping = tacotron_symbol_mapping()
    #print(json.dumps(mapping))
import os
import argparse
import subprocess

from munch import munchify

import ljspeech as Dataset

hp = munchify({
    'dataset': 'LJSpeech',
    'data_path': 'E:\\audio\\LJSpeech-1.1',
    'preprocessed_path': 'E:\\audio\\LJSpeech-1.1\\fastspeech_preprocessed',
    'mfa_path': 'D:\\montreal-forced-aligner',
    'mfa_align_path': 'D:\\montreal-forced-aligner\\bin\\mfa_align.exe',
    'hop_length': 256,
    'sampling_rate': 22050,
    'max_seq_len': 1000,
})

class Preprocessor():
    def __init__(self, args):
        self.args = args
        self.in_dir = hp.data_path
        self.out_dir = hp.preprocessed_path
        self.mfa_path = hp.mfa_path
        self.mfa_align_path = hp.mfa_align_path

    def exec(self):
        self.print_message()
        self.make_output_dirs(force=True)
        if self.args.prepare_align:
            print("Preparing alignment text data...")
            self.prepare_align()

        if self.args.mfa:
            print("Performing Montreal Force Alignment...")
            self.mfa()

        if self.args.create_dataset:
            print("Creating Training and Validation Dataset...")
            self.create_dataset()

    def print_message(self):
        print("\n")
        print("------ Preprocessing ------")
        print(f"* Data path   : {self.in_dir}")
        print(f"* Output path : {self.out_dir}")
        print("\n")
        print("The following will be executed:")
        # print("\n")
        if self.args.prepare_align:
            print("\t* Preparing Alignment Data")
        if self.args.mfa:
            print("\t* Montreal Force Alignmnet")
        if self.args.create_dataset:
            print("\t* Creating Training Dataset")
        print("\n")

    def make_output_dirs(self, force=True):
        out_dir = self.out_dir
        if self.args.mfa:
            mfa_out_dir = os.path.join(out_dir, "TextGrid")
            os.makedirs(mfa_out_dir, exist_ok=force)

        mel_out_dir = os.path.join(out_dir, "mel")
        os.makedirs(mel_out_dir, exist_ok=force)

        ali_out_dir = os.path.join(out_dir, "alignment")
        os.makedirs(ali_out_dir, exist_ok=force)

        f0_out_dir = os.path.join(out_dir, "f0")
        os.makedirs(f0_out_dir, exist_ok=force)

        energy_out_dir = os.path.join(out_dir, "energy")
        os.makedirs(energy_out_dir, exist_ok=force)

    ### Preprocessing ###
    def create_dataset(self):
        '''
        1. train and val meta will be obtained here
        2. during "build_fron_path", alignment, f0, energy and mel data will be created
        '''
        in_dir = self.in_dir
        out_dir = self.out_dir

        train, val = Dataset.build_from_path(in_dir, out_dir)
        with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            for m in train:
                f.write(m + '\n')
        with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
            for m in val:
                f.write(m + '\n')

    ### Prepare Align
    def prepare_align(self):
        in_dir = self.in_dir
        Dataset.prepare_align(in_dir)

    ### MFA ###
    def mfa(self):
        in_dir = self.in_dir
        out_dir = self.out_dir
        mfa_path = self.mfa_path
        mfa_bin_path = self.mfa_align_path

        mfa_out_dir = os.path.join(out_dir, "TextGrid")
        mfa_pretrain_path = os.path.join(mfa_path, "pretrained_models", "librispeech-lexicon.txt")
        cmd = f"{mfa_bin_path} {in_dir} {mfa_pretrain_path} english {mfa_out_dir} -j 8"
        print(f"Executing {cmd}")
        result = subprocess.Popen(cmd, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)
        assert result == 0


def main(args):
    P = Preprocessor(args)
    P.exec()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare_align', action="store_true", default=False)
    parser.add_argument('--mfa', action="store_true", default=False)
    parser.add_argument('--create_dataset', action="store_true", default=False)
    args = parser.parse_args()

    main(args)
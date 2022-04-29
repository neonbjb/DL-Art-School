import argparse
import os
import shutil
from subprocess import Popen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to search for files')
    parser.add_argument('--output_path', type=str, help='Path for output files')
    args = parser.parse_args()

    cmds = [
        f"scripts/audio/preparation/phase_1_split_files.py --path={args.path} --progress_file={args.output_path}_t1/progress.txt --num_threads=6 --output_path={args.output_path}_t1",
        f"scripts/audio/preparation/phase_2_sample_and_filter.py --path={args.output_path}_t1 --progress_file={args.output_path}/progress.txt --num_threads=6 --output_path={args.output_path}",
        f"scripts/audio/preparation/phase_3_generate_similarities.py --path={args.output_path} --num_workers=4",
    ]
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.output_path + "_t1", exist_ok=True)

    for cmd in cmds:
        p = Popen("python " + cmd)
        p.wait()

    shutil.rmtree(args.output_path + "_t1")

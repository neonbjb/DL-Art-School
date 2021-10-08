import os
import shutil
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='in', type=str)
    parser.add_argument('basis', metavar='basis', type=str)
    parser.add_argument('garbage', metavar='garbage', type=str)
    args = parser.parse_args()
    print(f"Moving files from {args.input} to {args.garbage}")
    os.makedirs(args.garbage, exist_ok=True)

    with open(args.input) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            assert args.basis in line
            movefile = os.path.join(args.garbage, line.replace(args.basis, '')[1:])
            print(f'{line} -> {movefile}')
            os.makedirs(os.path.dirname(movefile), exist_ok=True)
            shutil.move(line, movefile)

    
    
    

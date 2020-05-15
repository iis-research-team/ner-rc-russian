from tqdm import tqdm 
import random
import argparse
import sys
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, type=str, required=True)
    parser.add_argument('--output_dir', default=None, type=str, required=True)
    parser.add_argument('--min_sentences', default=5, type=int)
    parser.add_argument('--train_split', default=0.9, type=float)
    parser.add_argument('--min_len', default=32, type=int)
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--use_low_cased', default=True, type=bool)
    
    args = parser.parse_args(sys.argv[1:])

    with open(args.input) as fin, open(os.path.join(args.output_dir, "wikitext_train.raw"), "w") as train, \
                                  open(os.path.join(args.output_dir, "wikitext_test.raw"), "w") as test:
        texts = []
        text = []
        i = 0
        for line in tqdm(fin.readlines()):
            if line == "\n":
                if len(text) >= args.min_sentences or len(" ".join(text)) > args.min_sentences * args.min_len :
                    print(i, len(text))
                    i += 1
                    texts.append(" ".join(text))
                    text = []
                    continue
            if args.max_len > len(line) > args.min_len and (line[0].isupper() or args.use_low_cased):
                text.append(line)
        print(len(texts))
        for i, text in tqdm(enumerate(texts)):
            r = random.random()
            if r < args.train_split:
                train.write(f"\n = {i} = \n\n{text}")
            else:
                test.write(f"\n = {i} = \n\n{text}")



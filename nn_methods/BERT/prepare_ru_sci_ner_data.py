import os
import argparse
import sys
import pandas as pd
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="./preprocessed")
    parser.add_argument('--output_dir', default="./ner_data")
    parser.add_argument('--test_split', default=0.2, type=float)
    parser.add_argument('--seed', default=42, type=float)

    args = parser.parse_args(sys.argv[1:])

    with open(os.path.join(args.output_dir, "train.txt"), "w", encoding="utf-8") as train_out, \
         open(os.path.join(args.output_dir, "dev.txt"), "w", encoding="utf-8") as test_out:
        examples = []
        for f_name in os.listdir(args.input_dir):
            if f_name == "relations.csv":
                continue

            print(f_name)
            df = pd.read_csv(os.path.join(args.input_dir, f_name))
            queries = df[["token", "reviewed"]].dropna().values
            example = []
            for q in queries:
               example.append(q)
               if q[0] == ".":
                   examples.append(example)
                   example = []
            if len(example) > 0:
                examples.append(example)

        random.seed(args.seed)
        for ex in examples:
            r = random.random()
            if r < args.test_split:
                for line in ex:
                    test_out.write(f"{line[0]} {line[1]}\n")
                test_out.write("\n")
            else:
                for line in ex:
                    train_out.write(f"{line[0]} {line[1]}\n")
                train_out.write("\n")

        labels = set()
        for ex in examples:
            for line in ex:
                labels.add(line[1])
        with open(os.path.join(args.output_dir, "labels.txt"), "w") as l_file:
            for label in labels:
                l_file.write(label)
                l_file.write("\n")

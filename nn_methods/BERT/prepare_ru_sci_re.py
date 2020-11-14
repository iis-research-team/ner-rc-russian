import os
import argparse
import sys
import pandas as pd
import random
from sklearn.model_selection import train_test_split


def find_entity_end(tokens, tags, start):
    end = start
    for i in range(start + 1, len(tokens)):
        if tags[i].startswith("B") or tags[i].startswith("O"):
            break
        end = i
    return end


def find_sentence_boundary(tokens, start, direction=-1):
    sent_start = start
    while 0 <= start < len(tokens):
        if tokens[start] in [".", "?"] and (start + 1 < len(tokens) and tokens[start + 1][0].isupper() or True):
            if direction == 1:
                sent_start = start
            break
        sent_start = start
        start += direction
    return sent_start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_input_dir', default="./raw_markup")
    parser.add_argument('--input', default="./raw_markup/relations_2.csv")
    parser.add_argument('--output_dir', default="./")
    parser.add_argument('--test_split', default=0.1, type=float)

    args = parser.parse_args(sys.argv[1:])

    with open(os.path.join(args.output_dir, "train.txt"), "w", encoding="utf-8") as train_out, \
            open(os.path.join(args.output_dir, "dev.txt"), "w", encoding="utf-8") as test_out:
        re_markup = pd.read_csv(args.input)
        re_markup["join"] = re_markup[["Marker1", "Marker2", "Marker3"]].fillna("").apply(
            lambda xs: "; ".join(list(set([a.strip() for x in xs for a in x.split("; ") if x]))), 1)

        examples = []
        err = 0
        real_names = os.listdir(args.ner_input_dir)
        for f_name, rels in re_markup[["Файл", "join"]].values:
            print(f_name)
            rels = rels.replace(": ", "; ").strip()
            if rels.endswith(";"):
                rels = rels[:-1]
            formatted_rels = [(x[:x.find("(")], int(x[x.find("(") + 1:x.find(":")]), int(x[x.find(":") + 1:-1])) for x in
                             rels.split("; ") if x]

            real_name = f_name + ".csv"
            df = pd.read_csv(os.path.join(args.ner_input_dir, real_name))

            for rel in formatted_rels:
                rel = (rel[0].strip(), rel[1], rel[2])
                example = ""
                ent = 0

                tokens = df["token"].dropna().values
                tags = df["reviewed"].dropna().values if "reviewed" in df.columns else df["Лена"].dropna().values

                ent1_end = find_entity_end(tokens, tags, rel[1])
                ent2_end = find_entity_end(tokens, tags, rel[2])
                sent_start = find_sentence_boundary(tokens, min(rel[1], rel[2]))
                sent_end = find_sentence_boundary(tokens, min(rel[1], rel[2]), 1)
                if rel[1] < rel[2]:
                    example = " ".join(tokens[sent_start:rel[1]]) + " <e1>" + " ".join(tokens[rel[1]:ent1_end + 1]) + \
                              "</e1> " + " ".join(tokens[ent1_end + 1:rel[2]]) + " <e2>" + \
                              " ".join(tokens[rel[2]:ent2_end + 1]) + "</e2> " + " ".join(tokens[ent2_end + 1:sent_end + 1])
                else:
                    example = " ".join(tokens[sent_start:rel[2]]) + " <e2>" + " ".join(tokens[rel[2]:ent2_end + 1]) + \
                              "</e2> " + " ".join(tokens[ent2_end + 1:rel[1]]) + " <e1>" + \
                              " ".join(tokens[rel[1]:ent1_end + 1]) + "</e1> " + " ".join(tokens[ent1_end + 1:sent_end + 1])
                example = example.strip().replace(" ,", ",").replace(" .", ".")
                tmp = 1

                if example:
                    examples.append((example, rel[0]))
                else:
                    print("No example")
                # print(examples[-1])
                # input()
                if examples[-1][0].find("<e1>") < 0 or examples[-1][0].find("</e1>") < 0 or \
                        examples[-1][0].find("<e2>") < 0 or examples[-1][0].find("</e2>") < 0:
                    print("Errore", rel, real_name)
                    print(examples[-1])
                    print("---------------------")
                    examples = examples[:-1]
                    err += 1
        print(err, len(examples))
        # 50

        X = [ex[0] for ex in examples]
        y = [ex[1] for ex in examples]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split, random_state=42, stratify=y)

        for label in ["USAGE", "PARTOF", "SYNONYMS", "ISA", "COMPARE", "NONE", "CAUSE"]:
            print(f"Train, {label} - {len([i for i in y_train if i == label])}, {len([i for i in y_train if i == label]) / len(y_train)}%")
        print("-------------------------------------")
        for label in ["USAGE", "PARTOF", "SYNONYMS", "ISA", "COMPARE", "NONE", "CAUSE"]:
            print(f"Test, {label} - {len([i for i in y_test if i == label])}, {len([i for i in y_test if i == label]) / len(y_test)}%")

        for i in range(len(X_train)):
            train_out.write(f"{X_train[i]}\t{y_train[i]}\n")
        for i in range(len(X_test)):
            test_out.write(f"{X_test[i]}\t{y_test[i]}\n")

        labels = set()
        for ex in examples:
            labels.add(ex[1])
        labels.add("NONE")
        with open(os.path.join(args.output_dir, "labels.txt"), "w") as l_file:
            for label in labels:
                l_file.write(label)
                l_file.write("\n")

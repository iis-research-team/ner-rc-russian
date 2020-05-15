import os
import argparse
import sys
import pandas as pd
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_input_dir', default="./")
    parser.add_argument('--input', default="./relations.txt")
    parser.add_argument('--output_dir', default="./")
    parser.add_argument('--test_split', default=0.1, type=float)

    args = parser.parse_args(sys.argv[1:])

    with open(os.path.join(args.output_dir, "train.txt"), "w") as train_out, \
         open(os.path.join(args.output_dir, "dev.txt"), "w") as test_out:
        re_markup = pd.read_csv(args.input, delimiter="\t")
        #print(re_markup.columns)
        #print(re_markup.head())
        re_markup["join"] = re_markup[["Настя", "Денис", "Лена"]].fillna("").apply(lambda xs: "; ".join(list(set([a.strip() for x in xs for a in x.split("; ") if x]))), 1)

        examples = []
        err = 0
        real_names = os.listdir(args.ner_input_dir)
        for f_name, rels in re_markup[["Файл", "join"]].values:
            #print(rels)
            rels = rels.replace(": ", "; ").strip()
            if rels.endswith(";"):
                rels = rels[:-1]
            formated_rels = [(x[:x.find("(")], int(x[x.find("(") + 1:x.find(":")]), int(x[x.find(":") + 1:-1])) for x in rels.split("; ") if x]
            #print(formated_rels)
            #input()

            real_name = ""
            for n in real_names:
                if n.startswith(f_name):
                    real_name = n
                    break
            assert real_name != ""
            df = pd.read_csv(os.path.join(args.ner_input_dir, real_name), delimiter="\t")

            for rel in formated_rels:
                #print(rel)
                rel = (rel[0].strip(), rel[1], rel[2])
                example = ""
                ent = 0
                #print(df.columns, real_name)
                #if "id" not in df.columns:
                #    print("Fuck!")
                for tok_id, token, tag in df[["id", "token", "Лена"]].dropna().values:
                    if ent != 0 and tag == "O":
                        example = example[:-1] + f"</e{ent}> "
                        if token == ".":
                            example = example[:-1] + token
                            examples.append((example, rel[0]))
                            example = ""
                            break
                        else:
                            example += token + " "
                        ent = 0
                        continue
                              
                    if tok_id < min(rel[1], rel[2]):
                        if token == ".":
                            example = ""
                        else:
                            example += token + " "
                    elif tok_id > max(rel[1], rel[2]):
                        if token == "." and ent == 0:
                            example = example[:-1] + token
                            examples.append((example, rel[0]))
                            example = ""
                            break
                        else:
                            example += token + " "
                    elif tok_id == rel[1]:
                        if ent != 0:
                            example = example[:-1] + "</e2> "
                        ent = 1
                        example += "<e1>" + token + " "
                    elif tok_id == rel[2]:
                        if ent != 0:
                            example = example[:-1] + "</e1> "
                        ent = 2
                        example += "<e2>" + token + " "
                    elif ent == 0 and token:
                        example += token + " "
                       
                                
                        
                if example:
                    if ent != 0:
                        example += f"</e{ent}>"
                    example = example[:-1] + "."
                    examples.append((example, rel[0]))
                #print(examples[-1])
                #input()
                if examples[-1][0].find("<e1>") < 0 or examples[-1][0].find("</e1>") < 0 or \
                    examples[-1][0].find("<e2>") < 0 or examples[-1][0].find("</e2>") < 0:
                    print("Errore", rel, real_name)
                    print(examples[-1])
                    print("---------------------")
                    examples = examples[:-1]
                    err += 1
        print(err, len(examples))
        # 50

        #input()


        #print(examples)

        for ex in examples:
            r = random.random()
            assert ex[0].find("<e1>") >= 0
            assert ex[0].find("</e1>") >= 0
            assert ex[0].find("<e2>") >= 0
            assert ex[0].find("</e2>") >= 0
            if r < args.test_split:
                test_out.write(f"{ex[0]}\t{ex[1]}\n")
                #test_out.write("\n")
            else:
                train_out.write(f"{ex[0]}\t{ex[1]}\n")
                #train_out.write("\n")

        labels = set()
        for ex in examples:
            labels.add(ex[1])
        labels.add("NONE")
        with open(os.path.join(args.output_dir, "labels.txt"), "w") as l_file:
            for label in labels:
                l_file.write(label)
                l_file.write("\n")


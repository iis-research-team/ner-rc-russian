import os
import argparse
import sys
import re
from nltk.tokenize import wordpunct_tokenize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default="./",
                    help='Path to brat-formatted data')
    parser.add_argument('-o', '--output', default="./",
                    help='File to output BIO-formatted data')
    parser.add_argument('-d', '--delimiter', default="\t",
                    help='Inner sentence delimiter')
    parser.add_argument('-t', '--tag', default="T",
                    help='Tag for query with entity info')
    

    args = parser.parse_args(sys.argv[1:])

    with open(args.output, "w") as out:
        common = []

        for filename in os.listdir(args.input):
            if not filename.endswith(".ann"):
                continue
            with open(os.path.join(args.input, filename.replace(".ann", ".txt")), "r") as txt, \
                     open(os.path.join(args.input, filename), "r") as ann:
                text = "".join(txt.readlines())
                tokens = wordpunct_tokenize(text)
                new_tokens = []
                for i in range(len(tokens)):
                    if re.match(r'^\W+$', tokens[i]):
                        for x in tokens[i]:
                            new_tokens.append(x)
                    elif re.match(r'\w+_+', tokens[i]):
                        new_tokens.append(tokens[i][:tokens[i].find("_")])
                        new_tokens.append(tokens[i][tokens[i].find("_"):])
                    else:
                        new_tokens.append(tokens[i])
                tokens = new_tokens
                entities = [line.strip().split(args.delimiter)[1].split(" ") for line in ann.readlines() if line.startswith(args.tag)]
                entities.sort(key=lambda x: int(x[1]))

                token2place = []
                pos = 0
                for token in tokens:
                    while text[pos:pos + len(token)] != token:
                        pos += 1
                    token2place.append([pos, pos + len(token)])
                    pos += len(token)

                if len(token2place) != len(tokens):
                    print("ERRRRR")
                    input()

                bio = []
                cur_entity = 0
                for i in range(len(tokens)):
                    if cur_entity == len(entities):
                        bio.append((tokens[i], "O"))
                        continue
                    ent_s, ent_e = entities[cur_entity][1:]
                    ent_s, ent_e = int(ent_s), int(ent_e)
                    tok_s, tok_e = token2place[i]
                    if tok_s < ent_s:
                        bio.append((tokens[i], "O"))
                    elif ent_s == tok_s and tok_e <= ent_e:
                        bio.append((tokens[i], "B-" + entities[cur_entity][0]))
                        if tok_e == ent_e:
                            cur_entity += 1
                    elif ent_s < tok_s and tok_e <= ent_e:
                        bio.append((tokens[i], "I-" + entities[cur_entity][0]))
                        if tok_e == ent_e:
                            cur_entity += 1
                    else:
                        print("Wrong")
                        input()

                common.extend(bio)

        sents = []
        sent = []
        for word in common:
            if word[0] == ".":
                sent.append(word)
                sents.append(sent)
                sent = []
            else:
                sent.append(word)
        if len(sent) > 0:
            sents.append(sent)

        for sent in sents:
            out.write("\n".join([" ".join(x) for x in sent]))
            out.write("\n\n")


import os
import argparse
import sys
import re
from razdel import sentenize

class Substring():
    def __init__(self, start, stop, text):
        self.start = start
        self.stop = stop
        self.text = text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default="./",
                    help='Path to brat-formatted data')
    parser.add_argument('-o', '--output', default="./",
                    help='File to output SemEvam2010-formatted data')
    parser.add_argument('-d', '--delimiter', default="\t",
                    help='Inner sentence delimiter')
    

    args = parser.parse_args(sys.argv[1:])

    with open(args.output, "w", encoding="utf-8") as out:
        common = []

        for filename in os.listdir(args.input):
            if not filename.endswith(".ann"):
                continue
            with open(os.path.join(args.input, filename.replace(".ann", ".txt")), "r") as txt, \
                     open(os.path.join(args.input, filename), "r") as ann:
                text = "".join(txt.readlines())
                 
                entities = {}
                relations = {}
                for line in ann.readlines():
                    if line.startswith("T"):
                        e_id, e_range = line.strip().split(args.delimiter)[:2]
                        e_range = e_range.split(" ")[1:]
                        entities[e_id] = e_range
                    elif line.startswith("R"):
                        r_id, r_args = line.strip().split(args.delimiter)
                        r_args = r_args.split(" ")
                        r_args[1] = r_args[1][5:]
                        r_args[2] = r_args[2][5:]
                        relations[r_id] = r_args

                sents = list(sentenize(text)) # start, stop, text
                
                data = []
                for r_id in relations:
                    tag, e1, e2 = relations[r_id]
                    e1_s, e1_e, = entities[e1]
                    e2_s, e2_e, = entities[e2]
                    e1_s, e1_e, e2_s, e2_e = int(e1_s), int(e1_e), int(e2_s), int(e2_e)
                    sent = None
                    for s in sents:
                        if s.start <= min(e1_s, e2_s) and s.stop >= max(e1_e, e2_e):
                            sent = s
                    if sent is None:
                        sent = Substring(min(e1_s, e2_s), max(e1_e, e2_e), text[min(e1_s, e2_s):max(e1_e, e2_e)])
                    ex = sent.text
                    if e1_s < e2_s:
                        ex = ex[0:e2_s - sent.start] + "<e2>" + ex[e2_s - sent.start:e2_e - sent.start] + \
                             "</e2>" + ex[e2_e - sent.start:]
                        ex = ex[0:e1_s - sent.start] + "<e1>" + ex[e1_s - sent.start:e1_e - sent.start] + \
                             "</e1>" + ex[e1_e - sent.start:]
                    else:
                        ex = ex[0:e1_s - sent.start] + "<e1>" + ex[e1_s - sent.start:e1_e - sent.start] + \
                             "</e1>" + ex[e1_e - sent.start:]
                        ex = ex[0:e2_s - sent.start] + "<e2>" + ex[e2_s - sent.start:e2_e - sent.start] + \
                             "</e2>" + ex[e2_e - sent.start:]

                    if "\n" in ex:
                        ex = ex.replace(":\n", ": ").replace(";\n", "; ").replace(": \n", ": ").replace("; \n", "; ")
                        tmp = ex.split("\n")
                        for s in tmp:
                            if s.find("<e1>") != -1 and s.find("</e1>") != -1 and \
                               s.find("<e2>") != -1 and s.find("</e2>") != -1:
                                ex = s
                                break
                    if "\n" in ex:
                        ex = ex.replace("\n", " ")
                    ex = ex.strip().replace("\t", " ").replace("  ", " ")

                    if len(ex.split(" ")) > 127:
                        ex = ex[min(ex.find("<e1>"), ex.find("<e2>")):max(ex.find("</e1>"), ex.find("</e2")) + 5]
                    
                    data.append((ex, tag))
                common.extend(data)

        for sent, tag in common:
            out.write(sent + "\t" + tag + "\n")

        tags = set()
        for sent, tag in common:
            tags.add(tag)
        print(tags)


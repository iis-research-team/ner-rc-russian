import os

with open("sci_texts.txt", "w") as out:
    for f_name in os.listdir("./science_texts"):
        with open(os.path.join("./science_texts", f_name)) as inp:
            out.write(inp.read().replace("\n", " "))
            out.write("\n\n")

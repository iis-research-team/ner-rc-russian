import os
import pandas as pd

if __name__ == "__main__":
    relations = pd.read_csv(os.path.join("raw_markup", "Relations.csv"))
    real_names = os.listdir("raw_markup/")

    if not os.path.exists("preprocessed"):
        os.makedirs("preprocessed")

    for f_name in relations["Файл"].values:
        print(f_name)

        real_name = ""
        for n in real_names:
            if n.startswith(f_name):
                real_name = n
                break
        assert real_name != ""

        entities = pd.read_csv(os.path.join("raw_markup", real_name))

        for rel_source in ["reviewed"]:
            if pd.isna(relations[relations["Файл"] == f_name][rel_source].values[0]):
                continue
            rels = relations[relations["Файл"] == f_name][rel_source].values[0].replace(": ", "; ").strip()
            if rels.endswith(";"):
                rels = rels[:-1]
            formatted_rels = [(x[:x.find("(")], int(x[x.find("(") + 1:x.find(":")]), int(x[x.find(":") + 1:-1])) for x in
                              rels.split("; ") if x]

            ids = entities["id"].values
            for i in range(1, len(ids)):
                if ids[i] != ids[i - 1] + 1:
                    formatted_rels = [(rel[0], rel[1] + 1, rel[2] + 1) if rel[1] >= ids[i - 1] + 1 else (rel[0], rel[1], rel[2]) for rel in formatted_rels]

            relations.loc[relations["Файл"] == f_name, rel_source] = "; ".join([f"{rel[0]}({rel[1]}:{rel[2]})" for rel in formatted_rels])

        entities["id"] = range(len(entities["id"]))
        entities[["id", "token", "reviewed"]].to_csv(os.path.join("preprocessed", f_name + ".csv"), index=False)

    relations[["Файл", "reviewed"]].to_csv(os.path.join("preprocessed", "relations.csv"), index=False)

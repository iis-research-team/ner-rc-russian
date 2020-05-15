#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os


def read_df(path, file_names, template):
    return pd.concat([pd.read_csv(os.path.join(path, name)) for name in file_names if name.startswith(template)],
                     ignore_index=True)


def read_files(path):
    file_names = os.listdir(path)
    entity2id = read_df(path, file_names, "entity2id")
    relation2id = read_df(path, file_names, "relation2id")
    triplets = read_df(path, file_names, "triplets")
    alias_entity = read_df(path, file_names, "alias_entity")
    return [entity2id, relation2id, triplets, alias_entity]


def save_files(dest, files):
    for file in files:
        file[0].to_csv(path_or_buf="tmp.txt", index=False, header=False, sep=file[2])
        with open("tmp.txt", "r") as tmp, open(os.path.join(dest, file[1] + ".txt"), "w") as out:
            out.writelines(str(file[0].shape[0]) + "\n")
            out.writelines(tmp.readlines())
    os.remove("tmp.txt")


def frequency_filter(e_min, r_min, entity2id, relation2id, triplets, alias_entity):
    """
    Filtering with memory optimization
    """
    print("Start frequency filtering")

    # print(entity2id.shape, relation2id.shape, triplets.shape)
    # Remove relations and entities which label unknown
    triplets = triplets[triplets["rel_id"].isin(relation2id["id"])]
    triplets = triplets[triplets["e1_id"].isin(entity2id["id"]) & triplets["e2_id"].isin(entity2id["id"])]
    # print(entity2id.shape, relation2id.shape, triplets.shape)

    # Remove triplets with entities that are less common than e_min
    e1_counts = triplets.groupby(["e1_id"]).size().reset_index(name='n')
    e2_counts = triplets.groupby(["e2_id"]).size().reset_index(name='n')
    e1_filtered = e1_counts[e1_counts["n"] > e_min]["e1_id"]
    e2_filtered = e2_counts[e2_counts["n"] > e_min]["e2_id"]
    del e1_counts, e2_counts
    e_filtered = pd.unique(pd.concat([e1_filtered, e2_filtered]))
    del e1_filtered, e2_filtered
    triplets = triplets[triplets["e1_id"].isin(e_filtered) & triplets["e2_id"].isin(e_filtered)]
    del e_filtered
    # print(entity2id.shape, relation2id.shape, triplets.shape)

    # Remove triplets with relations that are less common than r_min
    rel_counts = triplets.groupby(["rel_id"]).size().reset_index(name='n')
    rel_filtered = rel_counts[rel_counts["n"] > r_min]["rel_id"]
    del rel_counts
    triplets = triplets[triplets["rel_id"].isin(rel_filtered)]
    del rel_filtered
    # print(entity2id.shape, relation2id.shape, triplets.shape)

    # Remove filtered relations and entities from entity2id and relation2id
    relation2id = relation2id[relation2id["id"].isin(triplets["rel_id"])]
    entity2id = entity2id[entity2id["id"].isin(triplets["e1_id"]) | entity2id["id"].isin(triplets["e2_id"])]
    alias_entity = alias_entity[alias_entity["id"].isin(entity2id["id"])]
    alias_entity = pd.concat([entity2id, alias_entity], ignore_index=True)
    # print(entity2id.shape, relation2id.shape, triplets.shape)

    return [entity2id, relation2id, triplets, alias_entity]


def prepare_data(t_split, v_split, entity2id, relation2id, triplets, alias_entity):
    assert t_split + v_split < 1, "Wrong arguments: train + validation must be less than 1"

    # Convert wikidata ids to numerical ids
    e2i = {v: k for k, v in dict(enumerate(entity2id["id"])).items()}
    r2i = {v: k for k, v in dict(enumerate(relation2id["id"])).items()}
    entity2id["qid"] = entity2id["id"].copy()
    entity2id["id"] = entity2id["id"].map(e2i)
    # entity2id["label"] = entity2id["label"].str.replace(" ", "_")
    qid2id = entity2id[["qid", "id"]]
    entity2qid = entity2id[["label", "qid"]]
    entity2id = entity2id[["label", "id"]]
    entity2id["label"] = entity2id["label"].str.replace(" ", "_")
    relation2id["id"] = relation2id["id"].map(r2i)
    relation2id["label"] = relation2id["label"].str.replace(" ", "_")
    relation2id = relation2id[["label", "id"]]
    triplets["e1_id"] = triplets["e1_id"].map(e2i)
    triplets["e2_id"] = triplets["e2_id"].map(e2i)
    triplets["rel_id"] = triplets["rel_id"].map(r2i)
    triplets = triplets[["e1_id", "e2_id", "rel_id"]]

    # Train/test/val split
    mask_array = np.random.rand(triplets.shape[0])
    test_mask = mask_array < t_split
    val_mask = mask_array > 1 - v_split

    test_triplets = triplets[["e1_id", "e2_id", "rel_id"]][test_mask]
    val_triplets = triplets[["e1_id", "e2_id", "rel_id"]][val_mask]
    train_triplets = triplets[["e1_id", "e2_id", "rel_id"]][~(test_mask | val_mask)]
    alias_entity = alias_entity[["label", "id"]]
    # print(test_triplets.shape, val_triplets.shape, train_triplets.shape)

    return [alias_entity, entity2id, entity2qid, qid2id, relation2id, train_triplets, test_triplets, val_triplets]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default=os.path.join(os.path.curdir, "extracted"), help="Path to extracted data from wikidata")
    parser.add_argument("-d", "--destination_path", default=os.path.join(os.path.curdir, "result"),
                        help="Path to extracted entities and relations")
    parser.add_argument("-e", "--entities_min", default=10, help="Number of minimum frequency of entities")
    parser.add_argument("-r", "--relations_min", default=10, help="Number of minimum frequency of relations")
    parser.add_argument("-t", "--testing", default=0.05, help="Percent of triplets for testing")
    parser.add_argument("-v", "--validation", default=0.05, help="Percent of triplets for validation")
    args = parser.parse_args()
    path, destination, e_min, r_min, t_split, v_split = args.path, args.destination_path, int(args.entities_min), \
                                                    int(args.relations_min), float(args.testing), float(args.validation)


    data = read_files(path)
    data = frequency_filter(e_min, r_min, *data)
    data = prepare_data(t_split, v_split, *data)

    files = zip(data,
                ["alias_entity", "entity2id", "entity_map", "qid2id", "relation2id", "train2id", "test2id", "valid2id"],
                ["\t", "\t", "\t", " ", " ", " ", " ", " "])

    save_files(destination, files)

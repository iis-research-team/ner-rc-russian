#!/usr/bin/env python3
import bz2
import json
import pandas as pd
import pydash
import argparse
import os
import traceback
from multiprocessing import Pool
from tqdm import tqdm


def wikidata(filename):
    with bz2.open(filename, mode='rt') as f:
        f.read(2)  # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError:
                continue


# 75586501 - all lines, 9.34.15 - time of void cycle
# 15000 tqdm iterations = 10000 real iterations
# 10000 lines, 1 process, time - 00:20:00, triplets - 4.3 Mb, entities -382 Kb


def run_proc(id, n, s, i2save, path, destination):
    i = 0
    entity2id = pd.DataFrame(columns=['id', 'label'])
    relation2id = pd.DataFrame(columns=['id', 'label'])
    triplets = pd.DataFrame(columns=['e1_id', 'e2_id', 'rel_id'])
    alias_entity = pd.DataFrame(columns=['label', 'id'])
    for record in tqdm(wikidata(path)):
        try:
            if i >= s and i % n == id:
                if record["id"][0] == "Q" and "ru" in record["labels"].keys():
                    # entity2id.loc[len(entity2id)] = {'id': record["id"], 'label': record["labels"]["ru"]["value"]}
                    new_ent = pd.DataFrame({'id': [record["id"]], 'label': [record["labels"]["ru"]["value"]]})
                    entity2id = entity2id.append(new_ent, ignore_index=True)
                    for claim in record["claims"]:
                        for rel in record["claims"][claim]:
                            if rel["mainsnak"]["datatype"] == "wikibase-item" and "datavalue" in rel["mainsnak"].keys():
                                # triplets.loc[len(triplets)] = {'e1_id': record["id"],
                                #                                'e2_id': rel["mainsnak"]["datavalue"]["value"]["id"],
                                #                                'rel_id': claim}
                                new_trip = pd.DataFrame({'e1_id': [record["id"]],
                                                         'e2_id': [rel["mainsnak"]["datavalue"]["value"]["id"]],
                                                         'rel_id': [claim]})
                                triplets = triplets.append(new_trip, ignore_index=True)
                    # for lang in ["ru", "en"]:
                    #     if lang in record["aliases"]:
                    #         for alias in record["aliases"][lang]:
                    #             # alias_entity.loc[len(alias_entity)] = {"label": val["value"], "id": record["id"]}
                    #             new_alias = {"label": [alias["value"]], "id": [record["id"]]}
                    #             alias_entity = alias_entity.append(new_alias, ignore_index=True)
                    for alias in record["aliases"]:
                        if alias == "ru" or alias == "en":
                            for val in record["aliases"][alias]:
                                # alias_entity.loc[len(alias_entity)] = {"label": val["value"], "id": record["id"]}
                                new_alias = pd.DataFrame({"label": [val["value"]], "id": [record["id"]]})
                                alias_entity = alias_entity.append(new_alias, ignore_index=True)
                elif record["id"][0] == "P" and "ru" in record["labels"].keys():
                    # relation2id.loc[len(relation2id)] = {"id": record["id"], "label": record["labels"]["ru"]["value"]}
                    new_rel = {"id": [record["id"]], "label": [record["labels"]["ru"]["value"]]}
                    relation2id = relation2id.append(new_rel, ignore_index=True)

            i += 1
            if i > s and i % i2save == 0:
                dest = os.path.join(destination, str(i))
                if not os.path.exists(dest):
                    os.mkdir(dest)
                pd.DataFrame.to_csv(entity2id, path_or_buf=os.path.join(dest, 'entity2id_' + str(id) + '.csv'))
                pd.DataFrame.to_csv(relation2id, path_or_buf=os.path.join(dest, 'relation2id_' + str(id) + '.csv'))
                pd.DataFrame.to_csv(triplets, path_or_buf=os.path.join(dest, 'triplets_' + str(id) + '.csv'))
                pd.DataFrame.to_csv(alias_entity, path_or_buf=os.path.join(dest, 'alias_entity_' + str(id) + '.csv'))
        except Exception:
            traceback.print_exc()
    dest = os.path.join(destination, "full")
    if not os.path.exists(dest):
        os.mkdir(dest)
    pd.DataFrame.to_csv(entity2id, path_or_buf=os.path.join(dest, 'entity2id_full' + str(id) + '.csv'))
    pd.DataFrame.to_csv(relation2id, path_or_buf=os.path.join(dest, 'relation2id_full' + str(id) + '.csv'))
    pd.DataFrame.to_csv(triplets, path_or_buf=os.path.join(dest, 'triplets_full' + str(id) + '.csv'))
    pd.DataFrame.to_csv(alias_entity, path_or_buf=os.path.join(dest, 'alias_entity_full' + str(id) + '.csv'))
    print('All items finished, final CSV exported!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default=os.path.join(os.path.curdir, "latest-all.json.bz2"), help="Path to wikidata")
    parser.add_argument("-d", "--destination_path", default=os.path.join(os.path.curdir, "extracted"),
                        help="Path to extracted entities and relations")
    parser.add_argument("-n", "--num_processes", default=16, help="Number of processes for multiprocessing")
    parser.add_argument("-s", "--start_index", default=0, help="Index of the line after which processing begins")
    parser.add_argument("-i", "--iter_to_save", default=10000000, help="Number of iterations after which saving")
    args = parser.parse_args()
    path, destination, n, s, i2save = args.path, args.destination_path, int(args.num_processes), \
                                      int(args.start_index), int(args.iter_to_save)
    if not os.path.exists(destination):
        os.mkdir(destination)
    p = Pool(n)
    for i in range(n):
        p.apply_async(run_proc, args=(i, n, s, i2save, path, destination))
    p.close()
    p.join()


""" 
TO DO: 
1. Remove index columns
2. Check line 47
"""

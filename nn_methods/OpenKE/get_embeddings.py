import json
import numpy as np
from openke.module.model import TransE
from openke.data import TrainDataLoader
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", default=os.path.join(os.path.curdir, "benchmarks"), 
                        help="Path to benchmark")
    parser.add_argument("-c", "--checkpoint", default=os.path.join(os.path.curdir, "checkpoint"),
                        help="Path to model checkpoint")
    parser.add_argument("-e", "--embedding", default=os.path.join(os.path.curdir, "kg_embed"), 
                        help="Path to saving embeddings")
   

    args = parser.parse_args()
    bench_path, ckpt_path, emb_path = args.benchmark, args.checkpoint, args.embedding

    # dataloader for training
    train_dataloader = TrainDataLoader(
            in_path = bench_path, 
            nbatches = 100,
            threads = 16, 
            sampling_mode = "normal", 
            bern_flag = 1, 
            filter_flag = 1, 
            neg_ent = 25,
            neg_rel = 0)

    # define the model
    transe = TransE(
            ent_tot = train_dataloader.get_ent_tot(),
            rel_tot = train_dataloader.get_rel_tot(),
            dim = 100, 
            p_norm = 1, 
            norm_flag = True)


    transe.load_checkpoint(os.path.join(ckpt_path, "transe.ckpt"))
    params = transe.get_parameters()
    np.savetxt(os.path.join(emb_path, "entity2vec.vec"), params["ent_embeddings.weight"])
    np.savetxt(os.path.join(emb_path, "relation2vec.vec"), params["rel_embeddings.weight"])

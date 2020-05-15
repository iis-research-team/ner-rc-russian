import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", default=os.path.join(os.path.curdir, "benchmarks"), 
                        help="Path to benchmark")
    parser.add_argument("-c", "--checkpoint", default=os.path.join(os.path.curdir, "checkpoint"),
                        help="Path to model checkpoint")
    parser.add_argument("-e", "--epochs", default=10000, help="Number of epochs")
    parser.add_argument("-a", "--alpha", default=0.001, help="Alpha hyperparameter")
    parser.add_argument("-t", "--test", default=False, help="Do testing")

    args = parser.parse_args()
    bench_path, ckpt_path, epochs, alpha, test = args.benchmark, \
                           args.checkpoint, int(args.epochs), float(args.alpha), \
                           bool(args.test)

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

    # dataloader for test
    if test:
        test_dataloader = TestDataLoader(bench_path, "link", type_constrain=False)

    # define the model
    transe = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 100, 
        p_norm = 1, 
        norm_flag = True)


    # define the loss function
    model = NegativeSampling(
        model = transe, 
        loss = MarginLoss(margin = 2.0),
        batch_size = train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=epochs, alpha=alpha, use_gpu=True)
    trainer.run()
    transe.save_checkpoint(os.path.join(ckpt_path, "transe.ckpt"))

    # test the model
    if test:
        transe.load_checkpoint(os.path.join(ckpt_path, "transe.ckpt"))
        tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
        tester.run_link_prediction(type_constrain=False)















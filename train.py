import argparse

import mlx.core as mx
from mlx.optimizers import AdamW

from data.vision import cifar10, mnist
from models import trm
from training.trainer import Trainer

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=["mnist", "cifar10"],
    help="dataset to use",
)
parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch size")
parser.add_argument("-e", "--epochs", type=int, default=15, help="number of epochs")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


def main(args):
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)

    if args.dataset == "mnist":
        train_data, test_data = mnist(args.batch_size)
    elif args.dataset == "cifar10":
        train_data, test_data = cifar10(args.batch_size)
    else:
        raise NotImplementedError(f"{args.dataset=} is not implemented.")
    n_inputs = next(train_data)["image"].shape[1:]
    train_data.reset()

    config = trm.ModelConfig(
        in_channels=n_inputs[-1],
        depth=1,
        dim=32,
        heads=4,
        patch_size=(7, 7),
        n_outputs=10,
    )
    model = trm.Model(config)
    model.summary()

    optimizer = AdamW(learning_rate=args.lr)

    manager = Trainer(model, optimizer)
    manager.train(train_data, val=test_data, epochs=args.epochs)

    #! plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    lw = 2
    ax.plot(mx.array(manager.train_acc_trace) * 100, label="train", color="r", lw=lw)
    ax.plot(mx.array(manager.val_acc_trace) * 100, label="val", color="b", lw=lw)
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(parser.parse_args())

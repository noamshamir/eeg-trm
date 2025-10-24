import argparse

import mlx.core as mx
import mlx.optimizers as optim

from data.vision import cifar10, mnist
from data.eeg import bcic_psd
from models import trm
from training.trainer import Trainer

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=["mnist", "cifar10", "eeg"],
    help="dataset to use",
)
parser.add_argument("-b", "--batch_size", type=int, default=1024, help="batch size")
parser.add_argument("-e", "--epochs", type=int, default=15, help="number of epochs")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


def main(args):
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)

    if args.dataset == "mnist":
        train_data, test_data, meta = mnist(args.batch_size)
    elif args.dataset == "cifar10":
        train_data, test_data, meta = cifar10(args.batch_size)
    elif args.dataset == "eeg":
        train_data, test_data, meta = bcic_psd(args.batch_size) 

    else:
        raise NotImplementedError(f"{args.dataset=} is not implemented.")
    n_inputs = next(train_data)["image"].shape[1:]
    train_data.reset()
    
    n_classes = 10 if args.dataset in ["mnist", "cifar10"] else meta.get("n_classes", 3)

    config = trm.ModelConfig(
        in_channels=n_inputs[-1],
        depth=2,
        dim=64,
        heads=4,
        patch_size=(4, 4),
        n_outputs=n_classes,
    )
    model = trm.Model(config)
    model.summary()

    n_steps = args.epochs * meta["steps_per_epoch"]
    n_linear = n_steps * 0.10
    linear = optim.linear_schedule(0, args.lr, steps=n_linear)
    cosine = optim.cosine_decay(args.lr, n_steps - n_linear, 0)
    lr_schedule = optim.join_schedules([linear, cosine], [n_linear])
    optimizer = optim.AdamW(
        learning_rate=lr_schedule, betas=(0.9, 0.999), weight_decay=0.01
    )

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

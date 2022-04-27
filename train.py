import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from  pytorch_msssim import MS_SSIM
from costumDataset import Kaiset
import sys
#chooses what model to train
if config.MODEL == "ResUnet":
    from resUnet import Generator
else:
    from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import localtime
import os
if not os.path.exists("evaluation"):
    os.mkdir("evaluation")
writer=SummaryWriter("train{}-{}-{}".format(localtime().tm_mon,localtime().tm_mday,localtime().tm_hour))
torch.backends.cudnn.benchmark = True


def train_fn(
     gen, loader, opt_gen, l1_loss,  g_scaler, epoch=0
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        if idx==0:
            xp = torch.zeros(x.shape).to(config.DEVICE)
        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x,xp)
            xp=y_fake.clone().detach()



        # Train generator
        with torch.cuda.amp.autocast():
            print()
            if sys.argv[2]=="L1":
                L1 = l1_loss(y_fake, y) * int(sys.argv[3])
            else:
                L1 = (1 - l1_loss((y_fake.type(torch.DoubleTensor) + 1) / 2, (y.type(torch.DoubleTensor) + 1) / 2)) * int(sys.argv[3])
            G_loss = + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            writer.add_scalar("L1 train loss",L1.item()/config.L1_LAMBDA,epoch*(len(loop))+idx)
            loop.set_postfix(

                L1    =L1.item()
            )
def test_fn(
    gen, loader, l1_loss, epoch=0
):
    loop = tqdm(loader, leave=True)

    gen.eval()
    with torch.no_grad():
     resultat=[]
     for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        if idx == 0:
            xp = torch.zeros(x.shape).to(config.DEVICE)
        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x, xp)
            xp = y_fake.clone().detach()




        # Train generator
        with torch.cuda.amp.autocast():


            if sys.argv[2] == "L1":
                L1 = l1_loss(y_fake, y) * int(sys.argv[3])
            else:
                L1 = (1 - l1_loss((y_fake.type(torch.DoubleTensor) + 1) / 2, (y.type(torch.DoubleTensor) + 1) / 2)) * int(sys.argv[3])
            G_loss = L1
            resultat.append(L1.item())



        if idx % 10 == 0:
            writer.add_scalar("L1 test loss",L1.item()/config.L1_LAMBDA,epoch*(len(loop))+idx)
            loop.set_postfix(

                L1    =L1.item()
            )
    gen.train()
    return torch.tensor(resultat).mean()
def main():
    #instancing the models
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    #print(disc)
    gen = Generator(init_weight=config.INIT_WEIGHTS).to(config.DEVICE)
    #print(gen)
    #instancing the optims
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    #instancing the Loss-functions
    BCE = nn.BCEWithLogitsLoss()
    if sys.argv[2]=="L1":
        L1_LOSS = nn.L1Loss()
    else:
        L1_LOSS = MS_SSIM(data_range=1, size_average=True, channel=3, win_size=11)

    #if true loads the checkpoit in the ./
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    #training data loading
    train_dataset = Kaiset(path=sys.argv[1], Listset=config.TRAIN_LIST)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    test_dataset = Kaiset(path=sys.argv[1],train=False, Listset=config.TRAIN_LIST)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=    False,
        num_workers=config.NUM_WORKERS,
    )
    #enabling MultiPrecision Mode, the optimise performance
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #evauation data loading

    best=10000000
    resultat=1
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            gen, train_loader, opt_gen, L1_LOSS,  g_scaler, epoch=epoch
        )
        resultat=test_fn( gen, test_loader,  L1_LOSS,  epoch=epoch)
        if best>resultat:
            print("improvement of the loss from {} to {}".format(best,resultat))
            best = resultat
        save_checkpoint(gen, opt_gen, epoch, filename=config.CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, epoch, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, test_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()

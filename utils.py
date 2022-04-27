import torch
import config
from torchvision.utils import save_image
import torch.nn as nn

#saves 16 images from a loader
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        if epoch == 0:
            save_image(x * 0.5 + 0.5, folder + f"/input.png")
            save_image(y * 0.5 + 0.5, folder + f"/label.png")
    gen.train()


def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, str(epoch) + filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

#initializes the wieghts using normal distro with mean 0 and std 0.02 (paper)
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)

import os
import sys
import time

import gc
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from tqdm import tqdm

import utils
from transformer_net import AutoEncoder
from vgg import Vgg16
import config


def train():
    device = torch.device("cuda" if config.CUDA else "cpu")

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # Transforms applied to Training Dataset
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda p: p.mul(255))
    ])
    train_dataset = datasets.ImageFolder(config.DATASET_PATH, transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE)

    transformer = AutoEncoder().to(device)  # Transformer
    optimizer = Adam(transformer.parameters(), config.LR)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16().to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda p: p.mul(255))
    ])
    style = utils.load_image(config.STYLE_IMAGE, size=config.STYLE_SIZE)
    style = style_transform(style)
    style = style.repeat(config.BATCH_SIZE, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(config.EPOCHS):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(tqdm(train_loader)):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = config.CONTENT_WEIGHT * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= config.STYLE_WEIGHT

            total_loss = content_loss + style_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % config.LOG_INTERVAL == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if config.CHECKPOINT_MODEL_DIR is not None and (batch_id + 1) % config.CHECKPOINT_INTERVAL == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(config.CHECKPOINT_MODEL_DIR, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(config.EPOCHS) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        config.CONTENT_WEIGHT) + "_" + str(config.STYLE_WEIGHT) + ".model"
    save_model_path = os.path.join(config.SAVE_MODEL_DIR, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def main():
    if config.CUDA and not torch.cuda.is_available():
        print('Cuda is not available.')

    gc.collect()
    torch.cuda.empty_cache()

    train()


if __name__ == "__main__":
    main()

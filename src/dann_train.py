import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math

def alpha_schedule(epoch, step, max_epoch=10, max_step=500):
    p = float(epoch * max_step + step) / (max_epoch * max_step)
    return 2. / (1. + math.exp(-10 * p)) - 1

def train_dann(
    feature_extractor, label_classifier, domain_discriminator,
    source_loader, target_loader, device,
    epochs=10, alpha_schedule=alpha_schedule
):
    feature_extractor.to(device)
    label_classifier.to(device)
    domain_discriminator.to(device)

    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_f = optim.Adam(feature_extractor.parameters(), lr=1e-3)
    optimizer_c = optim.Adam(label_classifier.parameters(), lr=1e-3)
    optimizer_d = optim.Adam(domain_discriminator.parameters(), lr=1e-3)

    max_step = min(len(source_loader), len(target_loader))

    for epoch in range(epochs):
        feature_extractor.train()
        label_classifier.train()
        domain_discriminator.train()

        tqdm_bar = tqdm(
            zip(source_loader, target_loader),
            desc=f"Epoch {epoch+1}",
            total=max_step
        )

        for step, ((src_x, src_y), (tgt_x, _)) in enumerate(tqdm_bar):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            batch_size = src_x.size(0)
            tgt_batch_size = tgt_x.size(0)

            # Объединяем source и target для GRL
            all_x = torch.cat([src_x, tgt_x], dim=0)

            alpha = alpha_schedule(epoch, step, max_epoch=epochs, max_step=max_step)

            features = feature_extractor(all_x)

            # Классификация (только на source)
            class_preds = label_classifier(features[:batch_size])
            loss_cls = cls_criterion(class_preds, src_y)

            # Domain discrimination
            domain_preds = domain_discriminator(features, alpha)
            domain_src = torch.ones(batch_size, 1).to(device)
            domain_tgt = torch.zeros(tgt_batch_size, 1).to(device)

            loss_domain = domain_criterion(domain_preds[:batch_size], domain_src) + \
                          domain_criterion(domain_preds[batch_size:], domain_tgt)

            loss = loss_cls + loss_domain

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            optimizer_d.zero_grad()

            loss.backward()

            optimizer_f.step()
            optimizer_c.step()
            optimizer_d.step()

            tqdm_bar.set_postfix({
                "cls_loss": loss_cls.item(),
                "domain_loss": loss_domain.item(),
                "alpha": round(alpha, 2)
            })

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def alpha_schedule(epoch, step, max_epoch=15, max_step=500, k=5.0):
    p = float(epoch * max_step + step) / (max_epoch * max_step)
    return 2. / (1. + np.exp(-k * p)) - 1

def train_dann(
    feature_extractor,
    label_classifier,
    domain_discriminator,
    source_loader,
    target_loader,
    device,
    epochs=10,
    alpha_schedule=None,
    lambda_domain=1.0,  # добавим вес
    batch_size = 128,
    evaluate_fn = None,
    target_test_loader = None,
):
    feature_extractor.to(device)
    label_classifier.to(device)
    domain_discriminator.to(device)

    optimizer_F = torch.optim.Adam(feature_extractor.parameters(), lr=1e-4)
    optimizer_C = torch.optim.Adam(label_classifier.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(domain_discriminator.parameters(), lr=1e-4)

    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    max_step = len(source_loader)

    target_accs = []  # сюда будем писать accuracy по target_test

    for epoch in range(epochs):
        feature_extractor.train()
        label_classifier.train()
        domain_discriminator.train()

        total_cls_loss = 0
        total_domain_loss = 0

        for step, ((src_x, src_y), (tgt_x, _)) in enumerate(tqdm(zip(source_loader, target_loader), total=max_step, desc=f"Epoch {epoch+1}")):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            src_bs = src_x.size(0)
            tgt_bs = tgt_x.size(0)
            all_x = torch.cat([src_x, tgt_x], dim=0)

            alpha = alpha_schedule(epoch, step, max_epoch=epochs, max_step=max_step)
            if step == 0:
                print(f"[Epoch {epoch+1}] alpha = {alpha:.3f}")

            features = feature_extractor(all_x)
            class_preds = label_classifier(features[:src_bs])
            domain_preds = domain_discriminator(features, alpha)

            domain_src = torch.ones((src_bs, 1)).to(device)
            domain_tgt = torch.zeros((tgt_bs, 1)).to(device)

            loss_cls = cls_criterion(class_preds, src_y)
            loss_domain = domain_criterion(domain_preds[:src_bs], domain_src) + \
                          domain_criterion(domain_preds[src_bs:], domain_tgt)

            loss = loss_cls + lambda_domain * loss_domain

            optimizer_F.zero_grad()
            optimizer_C.zero_grad()
            optimizer_D.zero_grad()

            loss.backward()
            optimizer_F.step()
            optimizer_C.step()
            optimizer_D.step()

            total_cls_loss += loss_cls.item()
            total_domain_loss += loss_domain.item()

        print(f"Epoch {epoch+1}: loss_cls={total_cls_loss/max_step:.4f} | loss_domain={total_domain_loss/max_step:.4f}")

        # evaluate на целевом тесте
        if evaluate_fn is not None and target_test_loader is not None:
            model = nn.Sequential(feature_extractor, label_classifier)
            acc = evaluate_fn(model, target_test_loader, device)
            print(f"Epoch {epoch+1}: target_test accuracy = {acc:.4f}")
            target_accs.append(acc)

    return target_accs

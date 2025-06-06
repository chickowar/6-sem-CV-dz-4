import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_dann(feature_extractor, label_classifier, domain_discriminator,
               source_loader, target_loader, device, epochs=10, alpha_schedule=None):

    feature_extractor.to(device)
    label_classifier.to(device)
    domain_discriminator.to(device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_f = optim.Adam(feature_extractor.parameters(), lr=1e-4)
    optimizer_c = optim.Adam(label_classifier.parameters(), lr=1e-4)
    optimizer_d = optim.Adam(domain_discriminator.parameters(), lr=1e-4)

    for epoch in range(epochs):
        feature_extractor.train()
        label_classifier.train()
        domain_discriminator.train()

        total_loss = 0
        tqdm_bar = tqdm(
            zip(source_loader, target_loader),
            desc=f"Epoch {epoch + 1}",
            total=min(len(source_loader), len(target_loader))
        )
        for i, ((src_x, src_y), (tgt_x, _)) in enumerate(tqdm_bar):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            batch_size = src_x.size(0)
            domain_src = torch.ones(batch_size, 1).to(device)
            domain_tgt = torch.zeros(batch_size, 1).to(device)

            x = torch.cat([src_x, tgt_x], dim=0)
            features = feature_extractor(x)

            # ---- Классификация меток ----
            preds_cls = label_classifier(features[:batch_size])
            loss_cls = class_criterion(preds_cls, src_y)

            # ---- Доменная дискриминация ----
            alpha = alpha_schedule(epoch, i) if alpha_schedule else 1.0
            domain_preds = domain_discriminator(features, alpha)
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

        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # Use raw logits for cross-entropy (apply softmax inside head during eval only)
        logits = model.backbone(images)
        logits = model.head.fc(logits) if hasattr(model.head, 'fc') else model.head.net(logits)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def eval_epoch(model, loader, device):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model.backbone(images)
            logits = model.head.fc(logits) if hasattr(model.head, 'fc') else model.head.net(logits)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

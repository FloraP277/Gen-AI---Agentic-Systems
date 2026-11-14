"""ResNet-18 classifier, fine-tuned from ImageNet weights. Needs a GPU (Colab T4 is enough)."""
import random

import numpy as np


def train_and_evaluate(train, val, test, classes, epochs: int = 8, seed: int = 0, batch: int = 64, lr: float = 3e-4):
    """Train on `train`, keep the epoch with the best validation balanced accuracy,
    return its predictions (class names) for `test`."""
    import torch
    from PIL import Image
    from sklearn.metrics import balanced_accuracy_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = {c: i for i, c in enumerate(classes)}
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1),
                                   transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])

    class Images(Dataset):
        def __init__(self, df, tf):
            self.paths, self.labels, self.tf = df["path"].tolist(), [idx[d] for d in df["dx"]], tf

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]

    loader = lambda df, tf, shuffle: DataLoader(Images(df, tf), batch_size=batch, shuffle=shuffle,  # noqa: E731
                                                num_workers=2, pin_memory=True)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    def predict(df):
        model.eval()
        out = []
        with torch.no_grad():
            for x, _ in loader(df, eval_tf, False):
                out += model(x.to(device)).argmax(1).cpu().tolist()
        return [classes[i] for i in out]

    best, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for x, y in loader(train, train_tf, True):
            opt.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        sched.step()
        score = balanced_accuracy_score(val["dx"], predict(val))
        print(f"epoch {epoch + 1}: val balanced accuracy {score:.3f}")
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(test)

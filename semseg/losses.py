import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


def bce2d_loss(input, target, reduction='mean'):
    B, H, W = target.shape
    input = input.reshape(B, H, W)
    assert (input.size() == target.size())
    target = target / 255  # transform
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


class UWCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1],
                 log_vars=torch.ones((2,), requires_grad=True), device="gpu:0") -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')
        self.log_vars = log_vars
        self.device = device

    def _forward(self, preds: Tensor, input: Tensor, labels: Tensor, label_bs: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        loss = torch.mean(loss_hard)
        loss_b = bce2d_loss(input, label_bs, reduction='mean')
        losses = [loss, loss_b]

        for i, log_var in enumerate(self.log_vars):
            log_var = log_var.to(self.device)
            losses[i] = (1 / 2) * (torch.exp(-log_var[0]) ** 2) * losses[i] + torch.log(torch.exp(log_var[0]) + 1)

        loss = sum(losses)
        return loss

    def forward(self, preds: Tensor, input: Tensor, labels: Tensor, label_bs: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, input, labels, label_bs) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, input, labels, label_bs)


class AUXCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1],
                 log_vars=torch.ones((2,), requires_grad=True), device="gpu:0") -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')
        self.log_vars = log_vars
        self.device = device

    def _forward(self, preds: Tensor, preds_aux: Tensor, input: Tensor, labels: Tensor, label_bs: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss_aux = self.criterion(preds_aux, labels).view(-1)
        loss_hard = loss_aux[loss_aux > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss_aux.topk(n_min)

        loss_aux = torch.mean(loss_hard)
        loss_b = bce2d_loss(input, label_bs, reduction='mean')

        losses = [loss_aux, loss_b]

        for i, log_var in enumerate(self.log_vars):
            log_var = log_var.to(self.device)
            losses[i] = (1 / 2) * (torch.exp(-log_var[0]) ** 2) * losses[i] + torch.log(torch.exp(log_var[0]) + 1)

        # loss of final seg
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        loss = torch.mean(loss_hard)
        losses.append(loss)
        # losses = [loss_aux, loss]
        loss = sum(losses)
        return loss

    def forward(self, preds: Tensor, preds_aux: Tensor, input: Tensor, labels: Tensor, label_bs: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, preds_aux, input, labels, label_bs) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, preds_aux, input, labels, label_bs)

__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice', 'UWCrossEntropy', 'AUXCrossEntropy']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


def get_uw_loss(loss_fn_name: str = 'UWCrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None,
                log_vars=torch.ones((2,), requires_grad=True), device="gpu:0"):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()

    return eval(loss_fn_name)(ignore_label, cls_weights, log_vars=log_vars, device=device)


def get_aux_loss(loss_fn_name: str = 'AUXCrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None,
                log_vars=torch.ones((2,), requires_grad=True), device="gpu:0"):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()

    return eval(loss_fn_name)(ignore_label, cls_weights, log_vars=log_vars, device=device)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)
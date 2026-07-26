import torch
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits

def cross_entropy_loss_with_logits(prediction, target, beta):
	label = target.long()
	mask = target.clone()
	num_positive = torch.sum(label == 1).float()
	num_negative = torch.sum(label == 0).float()

	mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
	mask[label == 0] = beta * num_positive / (num_positive + num_negative)
	mask[label == 2] = 0
	cost = binary_cross_entropy_with_logits(
		prediction, target, weight=mask, reduction='sum')
	return cost
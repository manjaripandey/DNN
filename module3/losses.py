import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-10

class KLDivLoss(nn.Module):
    # Standard KL divergence KL(p || q). p is the target (human) distribution, q is predicted.
    def forward(self, q, p):
        p = p.clamp(min=EPS)
        q = q.clamp(min=EPS)
        # sum p * (log p - log q)
        kl = (p * (p.log() - q.log())).sum(dim=1)
        return kl.mean()

class JSDLoss(nn.Module):
    # Jensen-Shannon Divergence. Symmetric and bounded version of KL. 
    def forward(self, q, p):
        p = p.clamp(min=EPS)
        q = q.clamp(min=EPS)
        m = 0.5 * (p + q)
        m = m.clamp(min=EPS)

        kl_pm = (p * (p.log() - m.log())).sum(dim=1)
        kl_qm = (q * (q.log() - m.log())).sum(dim=1)
        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        return jsd.mean()

class SoftCrossEntropyLoss(nn.Module):
    # Cross entropy using soft targets: -sum(p * log(q))
    def forward(self, q, p):
        p = p.clamp(min=EPS)
        q = q.clamp(min=EPS)
        ce = -(p * q.log()).sum(dim=1)
        return ce.mean()

class CustomLoss(nn.Module):
    # KL + squared difference in entropy. Forces the model to match the annotators' uncertainty level.
    def __init__(self, lambda_ent=1.0):
        super().__init__()
        self.lambda_ent = lambda_ent
        self.kl = KLDivLoss()

    def _entropy(self, probs):
        probs = probs.clamp(min=EPS)
        return -(probs * probs.log()).sum(dim=1)

    def forward(self, q, p):
        kl_term = self.kl(q, p)
        h_p = self._entropy(p)
        h_q = self._entropy(q)
        entropy_penalty = ((h_p - h_q) ** 2).mean()
        return kl_term + self.lambda_ent * entropy_penalty

class EMDLoss(nn.Module):
    # Simple Earth Mover's Distance using a cost matrix. Penalizes visually/semantically similar class confusion less.
    def __init__(self, distance_type='semantic'):
        super().__init__()
        self.distance_type = distance_type
        self.cost_matrix = self._build_cost_matrix()

    def _build_cost_matrix(self):
        n = 10
        C = torch.ones(n, n)

        if self.distance_type == 'semantic':
            for i in range(n):
                C[i, i] = 0.0
            
            # animal group: bird(2), cat(3), deer(4), dog(5), frog(6), horse(7)
            animals = [2, 3, 4, 5, 6, 7]
            for i in animals:
                for j in animals:
                    if i != j: C[i, j] = 0.5

            # vehicle group: plane(0), car(1), ship(8), truck(9)
            vehicles = [0, 1, 8, 9]
            for i in vehicles:
                for j in vehicles:
                    if i != j: C[i, j] = 0.5
        else:
            C = 1.0 - torch.eye(n)
        
        return C

    def forward(self, q, p):
        # Weighted L1 between distributions as an EMD proxy using broadcasting for (B, 10, 10)
        self.cost_matrix = self.cost_matrix.to(p.device)
        diff = (p.unsqueeze(2) - q.unsqueeze(1)).abs()
        cost = self.cost_matrix.unsqueeze(0)
        emd = (diff * cost).sum(dim=(1, 2))
        return emd.mean()

def get_loss_fn(name, cfg=None):
    # helper to pick loss function by name
    cfg = cfg or {}
    losses = {
        'KL': KLDivLoss(),
        'JSD': JSDLoss(),
        'SoftCE': SoftCrossEntropyLoss(),
        'Custom': CustomLoss(lambda_ent=cfg.get('lambda_entropy', 1.0)),
        'EMD': EMDLoss(distance_type=cfg.get('emd_distance_type', 'semantic')),
    }
    if name not in losses:
        raise ValueError(f"Unknown loss type: {name}")
    return losses[name]
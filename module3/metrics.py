import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine as cosine_distance

EPS = 1e-10

def compute_kl_divergence(p_true, q_pred):
    """Mean KL divergence across samples."""
    p = np.clip(p_true, EPS, 1.0)
    q = np.clip(q_pred, EPS, 1.0)
    
    kl = (p * (np.log(p) - np.log(q))).sum(axis=1)
    return kl.mean(), kl.std()

def compute_jsd(p_true, q_pred):
    """Jensen-Shannon Divergence."""
    p = np.clip(p_true, EPS, 1.0)
    q = np.clip(q_pred, EPS, 1.0)
    m = 0.5 * (p + q)
    m = np.clip(m, EPS, 1.0)
    
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=1)
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=1)
    jsd = 0.5 * (kl_pm + kl_qm)
    
    return jsd.mean(), jsd.std()

def compute_emd(p_true, q_pred, distance_type='semantic'):
    """Approximate Earth Mover's Distance using a cost matrix."""
    n_classes = 10
    C = np.ones((n_classes, n_classes))
    
    if distance_type == 'semantic':
        # Custom costs for semantically similar classes
        for i in range(n_classes):
            C[i, i] = 0.0
        
        animals = [2, 3, 4, 5, 6, 7]
        for i in animals:
            for j in animals:
                if i != j: C[i, j] = 0.5
        
        vehicles = [0, 1, 8, 9]
        for i in vehicles:
            for j in vehicles:
                if i != j: C[i, j] = 0.5
    else:
        C = 1.0 - np.eye(n_classes)
    
    n_samples = p_true.shape[0]
    emd_vals = []
    for i in range(n_samples):
        diff = np.abs(p_true[i:i+1, :].T - q_pred[i:i+1, :])
        emd = (C * diff.T).sum()
        emd_vals.append(emd)
    
    emd_vals = np.array(emd_vals)
    return emd_vals.mean(), emd_vals.std()

def compute_cosine_similarity(p_true, q_pred):
    """Average cosine similarity between distributions."""
    sims = []
    for i in range(p_true.shape[0]):
        s = 1.0 - cosine_distance(p_true[i], q_pred[i])
        sims.append(s)
    
    sims = np.array(sims)
    return sims.mean(), sims.std()

def compute_entropy(probs):
    """Calculates Shannon entropy."""
    probs = np.clip(probs, EPS, 1.0)
    return -(probs * np.log2(probs)).sum(axis=1)

def compute_entropy_correlation(p_true, q_pred):
    """Pearson and Spearman correlation between human and model entropy."""
    h_true = compute_entropy(p_true)
    h_pred = compute_entropy(q_pred)
    
    r_p, p_p = pearsonr(h_true, h_pred)
    r_s, p_s = spearmanr(h_true, h_pred)
    
    return r_p, p_p, r_s, p_s

def compute_precision_at_k(p_true, q_pred, ks=[100, 200, 500]):
    """Overlap between top-K high-entropy samples."""
    h_true = compute_entropy(p_true)
    h_pred = compute_entropy(q_pred)
    
    t_rank = np.argsort(-h_true)
    p_rank = np.argsort(-h_pred)
    
    res = {}
    for k in ks:
        top_t = set(t_rank[:k])
        top_p = set(p_rank[:k])
        res[k] = len(top_t & top_p) / k
    
    return res

def compute_sba(p_true, q_pred):
    """Soft-label Balanced Accuracy."""
    class_weights = p_true.mean(axis=0)
    class_errors = np.abs(p_true - q_pred).mean(axis=0)
    class_accs = 1.0 - class_errors
    
    return (class_weights * class_accs).sum() / class_weights.sum()

def evaluate_all_metrics(p_true, q_pred):
    """Runs all metrics and returns a results dictionary."""
    res = {}
    
    m, s = compute_kl_divergence(p_true, q_pred)
    res['kl_mean'], res['kl_std'] = m, s
    
    m, s = compute_jsd(p_true, q_pred)
    res['jsd_mean'], res['jsd_std'] = m, s
    
    m, s = compute_emd(p_true, q_pred)
    res['emd_mean'], res['emd_std'] = m, s
    
    m, s = compute_cosine_similarity(p_true, q_pred)
    res['cosine_mean'], res['cosine_std'] = m, s
    
    r_p, p_p, r_s, p_s = compute_entropy_correlation(p_true, q_pred)
    res['pearson_r'], res['pearson_p'] = r_p, p_p
    res['spearman_r'], res['spearman_p'] = r_s, p_s
    
    precs = compute_precision_at_k(p_true, q_pred)
    for k, v in precs.items():
        res[f'precision@{k}'] = v
    
    res['sba'] = compute_sba(p_true, q_pred)
    return res

def print_metrics(m, name="Model"):
    """Quick console print of results."""
    print(f"\n[{name}]")
    print(f"  KL:      {m['kl_mean']:.4f} ± {m['kl_std']:.4f}")
    print(f"  JSD:     {m['jsd_mean']:.4f} ± {m['jsd_std']:.4f}")
    print(f"  Cosine:  {m['cosine_mean']:.4f}")
    print(f"  Pearson: {m['pearson_r']:.4f}")
    print(f"  SBA:     {m['sba']:.4f}")
import torch
def dummy_input_multi_domain(num_domains, batch_size=10, emb_size=128):
    x = torch.randn(batch_size, emb_size)
    domain_ids = torch.randint(0, num_domains, size=(batch_size, ))
    return x, domain_ids

"""Check full-size forward/backward and optimizer step without dataset training."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/protocol_v1.json'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from src.protocol import read_protocol
    from src.experiment import build_model, normalize
    from src.train import FrequencyLoss, NegPearsonLoss
    config = read_protocol(args.config)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    for kind in ('physformer', 'bipulseformer'):
        torch.manual_seed(42)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = build_model(config, kind).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        x = torch.randn(config['batch_size'], 3, config['clip_len'], 128, 128, device=device)
        times = torch.arange(config['clip_len'], device=device) / config['fps']
        hr = torch.linspace(65, 95, config['batch_size'], device=device)
        labels = torch.sin(2 * torch.pi * times[None] * hr[:, None] / 60)
        pred = normalize(model(x, gra_sharp=config['gra_sharp'])[0])
        ce, kl = FrequencyLoss(config['fps'], config['bpm_low'], config['bpm_high'], diff_flag=True)(pred, hr)
        loss = config['alpha'] * NegPearsonLoss()(pred, labels) + config['beta'] * (ce + kl)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite synthetic loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            output = model(x[:1], gra_sharp=config['gra_sharp'])[0]
            if not torch.isfinite(output).all():
                raise FloatingPointError('Non-finite inference after optimizer step')
        print(json.dumps({'model': kind, 'shape': list(pred.shape), 'loss': loss.item(),
                          'gradient_norm': norm.item(),
                          'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20
                          if device.type == 'cuda' else None}), flush=True)
        del model, optimizer, x, labels, pred, ce, kl, loss, norm, output
        gc.collect()


if __name__ == '__main__':
    main()

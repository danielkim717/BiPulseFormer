import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.protocol import fingerprint, make_split, read_protocol
from src.models.bipulseformer import BiLevelRoutingAttention_TDC_gra_sharp, ViT_BiPulseFormer
from src.train import FrequencyLoss, estimate_hr_targets
from src.evaluation import evaluate_per_subject
from src.evaluation_per_clip import evaluate_per_clip

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = read_protocol(ROOT / 'configs/protocol_v1.json')

    def test_subject_splits_disjoint_reproducible(self):
        subjects = [f'{i:02d}' for i in range(1, 11)]
        for mode, sizes in [('intra', [6, 2, 2]), ('cross', [8, 2])]:
            split = make_split(subjects, mode, self.config)
            self.assertEqual(split, make_split(subjects[::-1], mode, self.config))
            self.assertEqual([len(s) for s in split.values()], sizes)
            flattened = sum(split.values(), [])
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertEqual(set(flattened), set(subjects))

    def test_config_hash_changes_with_condition(self):
        other = dict(self.config, lr=0.001)
        self.assertNotEqual(fingerprint(other), fingerprint(self.config))

    def test_aggregation_rejects_mixed_inventory_and_duplicate_seed(self):
        from scripts.summarize_runs import summarize
        paths = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, seed in enumerate([42, 43, 44]):
                folder = Path(tmp) / str(seed)
                folder.mkdir()
                (folder / 'config.json').write_text(json.dumps({'model': 'bipulseformer'}))
                report = {'seed': seed, 'config': self.config, 'split': {}, 'code_hash': 'same',
                          'sample_hashes': {}, 'test': {'per_clip': {}, 'per_recording': {}}}
                for key in ['MAE_bpm_clip', 'RMSE_bpm_clip', 'MAPE_pct_clip', 'Pearson_clip']:
                    report['test']['per_clip'][key] = i + 1
                for key in ['MAE_bpm', 'RMSE_bpm', 'MAPE_pct', 'Pearson', 'signal_Pearson_mean']:
                    report['test']['per_recording'][key] = i + 1
                path = folder / 'summary.json'
                path.write_text(json.dumps(report))
                paths.append(path)
            self.assertEqual(summarize(paths)['metrics']['per_clip']['MAE_bpm_clip']['mean'], 2)
            with self.assertRaises(ValueError):
                summarize([paths[0], paths[0], paths[2]])
            report['sample_hashes'] = {'test': 'different'}
            paths[-1].write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                summarize(paths)


class NumericsTests(unittest.TestCase):
    def test_full_routing_matches_baseline_model(self):
        from src.models.physformer_baseline import PhysFormer
        common = dict(dim=8, ff_dim=12, num_heads=2, num_layers=3, dropout_rate=0)
        torch.manual_seed(11)
        baseline = PhysFormer(**common).eval()
        routed = ViT_BiPulseFormer(**common, n_win=(1, 4, 4), topk=16,
                                   routing_mode='fft_magnitude', diff_routing=False).eval()
        routed.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(1, 3, 32, 128, 128)
        with torch.no_grad():
            reference = baseline(x)[0]
            actual = routed(x)[0]
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)

    def test_restoration_matches_evaluation_operator(self):
        from src.evaluation import _detrend
        torch.manual_seed(3)
        preds = torch.randn(2, 160, requires_grad=True)
        frequency = FrequencyLoss(diff_flag=True)
        frequency(preds, torch.tensor([70., 90.]))
        actual = preds.cumsum(-1) @ frequency._detrend_operator.T
        expected = np.stack([_detrend(np.cumsum(p), 100) for p in preds.detach().numpy().astype(np.float64)])
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-5, rtol=1e-4)

    def test_flat_prediction_loss_is_finite_and_backward(self):
        pred = torch.zeros(2, 160, requires_grad=True)
        ce, kl = FrequencyLoss()(pred, torch.tensor([72., 90.]))
        self.assertTrue(torch.isfinite(ce + kl))
        (ce + kl).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_hr_target_uses_loss_band(self):
        t = np.arange(160) / 30
        y = np.stack([np.sin(2 * np.pi * 1.2 * t), np.sin(2 * np.pi * 2 * t)])
        np.testing.assert_allclose(estimate_hr_targets(y, diff_flag=False), [72, 120], atol=1)
        with self.assertRaises(ValueError):
            estimate_hr_targets(np.zeros((1, 160)))

    def test_differentiated_harmonics_restored_for_target(self):
        t = np.arange(161) / 30
        raw = np.sin(2 * np.pi * 1.2 * t) + .65 * np.sin(2 * np.pi * 2.4 * t)
        derivative = np.diff(raw)[None]
        self.assertGreater(estimate_hr_targets(derivative, diff_flag=False)[0], 130)
        np.testing.assert_allclose(estimate_hr_targets(derivative), [72], atol=2)
        pred = torch.tensor(derivative, dtype=torch.float32, requires_grad=True)
        ce, kl = FrequencyLoss(diff_flag=True)(pred, torch.tensor([72.]))
        (ce + kl).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_pure_timestamp_alignment(self):
        from src.data.alignment import align_pure
        origin = 1392643993646759000
        entries = [{'Timestamp': origin + i * 100000000, 'Value': {'waveform': float(i)}}
                   for i in range(4)]
        paths = [f'Image{origin + i * 50000000}.png' for i in range(-1, 8)]
        kept, values, trimmed = align_pure(entries, paths)
        np.testing.assert_allclose(values, np.arange(7) * .5)
        self.assertEqual(len(kept), 7)
        self.assertEqual(trimmed, 2)

    def test_ste_and_sparse_forward_and_routing_gradient(self):
        torch.manual_seed(7)
        ste = BiLevelRoutingAttention_TDC_gra_sharp(8, 2, 0, .7,
                n_win=(1, 2, 2), topk=2, routing_mode='mean')
        sparse = copy.deepcopy(ste)
        sparse.diff_routing = False
        ste.train()
        sparse.train()
        x = torch.randn(2, 128, 8, requires_grad=True)
        actual, scores = ste(x, 2.)
        expected, _ = sparse(x, 2.)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        scores.retain_grad()
        actual.square().mean().backward()
        self.assertIsNotNone(scores.grad)
        self.assertGreater(scores.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_fft_token_fps_and_config_guard(self):
        model = ViT_BiPulseFormer(dim=8, ff_dim=12, num_heads=2, num_layers=3,
                                  n_win=(1, 4, 4), routing_mode='fft_magnitude')
        self.assertEqual(model.transformer1.blocks[0].attn.fps, 7.5)
        with self.assertRaises(ValueError):
            ViT_BiPulseFormer(num_layers=5)

    def test_ste_forward_matches_sparse_for_sharp_attention(self):
        torch.manual_seed(7)
        ste = BiLevelRoutingAttention_TDC_gra_sharp(8, 2, 0, .7,
                n_win=(1, 2, 2), topk=1, routing_mode='mean')
        sparse = copy.deepcopy(ste)
        sparse.diff_routing = False
        x = torch.randn(2, 128, 8)
        actual, _ = ste(x, .01)
        expected, _ = sparse(x, .01)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_evaluation_recordings_and_identical_signal(self):
        t = np.arange(240) / 30
        raw = np.sin(2 * np.pi * 1.2 * t)
        clips = np.stack([raw[:160], raw[80:240]])
        samples = [{'video_id': '01-01', 'first_frame_idx': i} for i in (0, 80)]
        metrics = evaluate_per_subject(clips, clips, samples, diff_flag=False)
        self.assertEqual(metrics['n_recordings'], 1)
        self.assertEqual(metrics['MAE_bpm'], 0)
        self.assertIsNone(metrics['RMSE_se'])
        self.assertAlmostEqual(metrics['signal_Pearson_mean'], 1, places=6)
        with self.assertRaises(ValueError):
            evaluate_per_subject(clips, clips, [])
        with self.assertRaises(ValueError):
            evaluate_per_clip(np.empty((0, 160)), np.empty((0, 160)))


class EngineTests(unittest.TestCase):
    def test_one_epoch_synthetic_training_test_only_after_validation(self):
        from src import experiment
        from torch.utils.data import DataLoader, Dataset
        events = []
        class FakeData(Dataset):
            def __init__(self, name, ids):
                self.dataset_name = name
                self.samples = [{'video_id': s, 'first_frame_idx': 0} for s in ids]
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, i):
                g = torch.Generator().manual_seed(i)
                x = torch.randn(3, 32, 128, 128, generator=g)
                y = torch.sin(torch.arange(32) * (2 * torch.pi * 1.5 / 30))
                return x, y
        def loader(name, root, ids, config, args, train=False):
            events.append('train' if train else name)
            return DataLoader(FakeData(name, ids), batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = read_protocol(ROOT / 'configs/protocol_v1.json')
            config.update(epochs=1, clip_len=32, eval_step=16, batch_size=2,
                          dim=8, ff_dim=12, num_heads=2, num_layers=3)
            path = tmp / 'config.json'
            path.write_text(json.dumps(config))
            args = SimpleNamespace(config=str(path), source='PURE', target='UBFC-rPPG',
                source_root='source', target_root='target', mode='cross', model='bipulseformer',
                seed=42, dry_run=False, output=str(tmp / 'run'), device='cpu', workers=0)
            original_eval = experiment.evaluate
            def checked_eval(*items):
                events.append('evaluate:' + items[1].dataset.dataset_name)
                return original_eval(*items)
            with patch.object(experiment, 'discover_subjects', return_value=['01', '02', '03', '04', '05']), \
                 patch.object(experiment, 'build_loader', side_effect=loader), \
                 patch.object(experiment, 'evaluate', side_effect=checked_eval):
                experiment.run(args)
                with self.assertRaises(FileExistsError):
                    experiment.run(args)
            summary = json.loads((tmp / 'run/summary.json').read_text())
            self.assertEqual(summary['best_epoch'], 1)
            self.assertEqual(events.count('evaluate:UBFC-rPPG'), 1)
            self.assertLess(events.index('evaluate:PURE'), events.index('UBFC-rPPG'))
            self.assertNotIn('test', summary['history'][0])
            state = torch.load(tmp / 'run/best.pt', weights_only=True)
            self.assertEqual(state['config']['clip_len'], 32)


if __name__ == '__main__':
    unittest.main()

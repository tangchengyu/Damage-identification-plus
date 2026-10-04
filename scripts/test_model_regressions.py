"""Regression checks using small tensors; no checkpoints or downloads required."""
import ast
import copy
import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sunshangshibie_tiaozhan/src'))
from predict import DamagePredictor


def lifetime_model_module():
    spec = importlib.util.spec_from_file_location('lifetime_architecture', ROOT / 'shoumingyuce/src/model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def early_stopping_class():
    path = ROOT / 'shoumingyuce/src/train.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'EarlyStopping')
    scope = {'copy': copy, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['EarlyStopping']


class ModelRegressionTests(unittest.TestCase):
    def test_lifetime_epoch_limit_still_saves_best_validation_weights(self):
        import os
        import json
        import time
        path = ROOT / 'shoumingyuce/src/train.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef))
                 and x.name in ('EarlyStopping', 'train_model', 'convert_numpy_types')]
        model = torch.nn.Linear(1, 1)
        metrics = dict(r2=0., rmse=1., mae=1., mape=1.)
        def train_epoch(model, loader, criterion, optimizer, device, epoch, writer):
            with torch.no_grad():
                model.weight.fill_(epoch+1)
            return float(epoch+1)
        def validate_epoch(model, *args):
            return model.weight.item(), metrics, [1.], [1.]
        scope = dict(os=os, json=json, time=time, torch=torch, copy=copy, np=__import__('numpy'),
                     nn=torch.nn, optim=torch.optim, train_epoch=train_epoch, validate_epoch=validate_epoch,
                     EarlyStopping=early_stopping_class(), SummaryWriter=MagicMock(),
                     create_lifetime_model=lambda *args: model,
                     create_lifetime_dataloaders=lambda **kwargs: ({'train': [], 'val': [], 'test': []},
                         {'train': 1,'val': 1,'test': 1}, dict(feature_scaler=None,lifetime_scaler=None,outlier_detector=None)),
                     plot_training_curves=lambda *args: None, plot_predictions_vs_targets=lambda *args: None)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        config = {'training': dict(batch_size=1, learning_rate=.001, weight_decay=0., num_epochs=2,
                                   early_stopping_patience=10),
                  'data': dict(train_ratio=.7,val_ratio=.15,test_ratio=.15)}
        with tempfile.TemporaryDirectory() as directory:
            trained, history, results = scope['train_model'](config, directory, directory)
            saved = torch.load(Path(directory) / 'models/best_lifetime_model.pth', weights_only=True)
            self.assertEqual(1., trained.weight.item())
            self.assertEqual(1., saved['model_state_dict']['weight'].item())
            self.assertEqual(1., results['test_loss'])

    def test_classification_loader_preserves_an_explicit_validation_split(self):
        import json
        from PIL import Image
        from data_loader import create_data_loaders
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            for split in ['train', 'val', 'test']:
                image = root / f'boluo/hongguan/{split}/1.jpg'
                image.parent.mkdir(parents=True)
                Image.new('RGB', (8, 8)).save(image)
                records[split] = {'path': f'data_boluo_hongguan_{split}_1', 'category-vector': [1,0,0,0]}
            (root / 'DATA.json').write_text(json.dumps({'boluo': records}), encoding='utf-8')
            loaders, sizes = create_data_loaders(directory, batch_size=1)
            self.assertEqual({'train': 1, 'val': 1, 'test': 1}, sizes)
            self.assertEqual('val', loaders['val'].dataset.dataset_items[0]['split'])

    def test_grouped_lifetime_split_has_no_specimen_overlap(self):
        import json
        from PIL import Image
        spec = importlib.util.spec_from_file_location('life_data_groups', ROOT / 'shoumingyuce/src/data_loader.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            for i in range(20):
                image = root / f'boluo/hongguan/train/{i}.jpg'
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (8, 8), (i*10, 0, 0)).save(image)
                records[str(i)] = {'path': f'data_boluo_hongguan_train_{i}', 'shuxing': {},
                                   'shouming': 100, 'specimen_id': f'gear-{i//2}',
                                   'lifetime_verified': True, 'lifetime_source': 'synthetic test fixture'}
            (root / 'DATA.json').write_text(json.dumps({'boluo': records}), encoding='utf-8')
            loaders, sizes, scalers = module.create_lifetime_dataloaders(directory, save_scalers=False)
            groups = {split: {x['specimen_id'] for x in loader.dataset.dataset_items}
                      for split, loader in loaders.items()}
            for a, b in [('train','val'), ('train','test'), ('val','test')]:
                self.assertTrue(groups[a].isdisjoint(groups[b]))
            self.assertEqual(sizes, {split: len(loader.dataset) for split, loader in loaders.items()})

    def test_selected_checkpoint_pair_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            for view in ['hongguan', 'tishijing']:
                (Path(directory) / f'model_{view}.pth').write_bytes(b'fixture')
            with self.assertRaises(FileNotFoundError):
                DamagePredictor(directory, device=torch.device('cpu'))

    def test_explicit_original_weights_remain_available(self):
        with tempfile.TemporaryDirectory() as directory:
            for view in ['hongguan', 'tishijing']:
                (Path(directory) / f'model_{view}.pth').write_bytes(b'fixture')
            with patch('predict.load_model', return_value=torch.nn.Identity()) as loader:
                predictor = DamagePredictor(directory, device=torch.device('cpu'), model_variant='original')
            self.assertEqual('original', predictor.model_variant)
            self.assertTrue(all('_updated' not in call.args[0] for call in loader.call_args_list))

    def test_lifetime_backbone_rejects_incomplete_checkpoint(self):
        module = lifetime_model_module()
        config = {'model': {'embedding_dim': 16, 'dropout': 0., 'num_heads': 2,
                            'num_layers': 1, 'hidden_dim': 32}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.pth'
            torch.save({'unrelated': torch.zeros(1)}, path)
            with self.assertRaisesRegex(ValueError, 'backbone'):
                module.create_lifetime_model(config, str(path), pretrained_backbone=False)

    def test_lifetime_loader_does_not_relabel_microscope_image_as_macro(self):
        import json
        from PIL import Image
        spec = importlib.util.spec_from_file_location('life_data', ROOT / 'shoumingyuce/src/data_loader.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for view in ['hongguan', 'tishijing']:
                image = root / 'boluo' / view / 'train' / '1.jpg'
                image.parent.mkdir(parents=True)
                Image.new('RGB', (8, 8)).save(image)
            metadata = {'boluo': {view: {'path': f'data_boluo_{view}_train_1',
                                        'shuxing': {}, 'shouming': 100} for view in ['hongguan', 'tishijing']}}
            (root / 'DATA.json').write_text(json.dumps(metadata), encoding='utf-8')
            self.assertEqual(['hongguan'], [row['sample_id'] for row in module.load_lifetime_dataset(directory)])

    def test_lifetime_dataset_prediction_respects_view_and_recorded_split(self):
        import json
        from PIL import Image
        # Load the lifetime predictor with its own module dependencies.
        path = ROOT / 'shoumingyuce/src/predict.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'LifetimePredictor')
        scope = {'os': __import__('os'), 'json': json}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), scope)
        predictor = scope['LifetimePredictor'].__new__(scope['LifetimePredictor'])
        calls = []
        predictor.predict_single = lambda path, attrs: calls.append(path) or 100
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'boluo/hongguan/train/1.jpg'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(image)
            records = {'micro': {'path': 'data_boluo_tishijing_train_1', 'shuxing': {}, 'shouming': 100},
                       'missing_test': {'path': 'data_boluo_hongguan_test_1', 'shuxing': {}, 'shouming': 200}}
            (root / 'DATA.json').write_text(json.dumps({'boluo': records}), encoding='utf-8')
            predictor.predict_from_dataset(directory)
            self.assertEqual([], calls)

    def test_lifetime_training_requires_verified_labels_and_specimen_groups(self):
        import json
        from PIL import Image
        spec = importlib.util.spec_from_file_location('life_data_training', ROOT / 'shoumingyuce/src/data_loader.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'boluo/hongguan/train/1.jpg'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(image)
            metadata = {'boluo': {'sample': {'path': 'data_boluo_hongguan_train_1',
                                            'shuxing': {}, 'shouming': 100}}}
            (root / 'DATA.json').write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'verified|specimen'):
                module.create_lifetime_dataloaders(directory, save_scalers=False)

    def test_default_uses_updated_pair_and_reports_loaded_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for view in ['hongguan', 'tishijing']:
                for suffix in ['', '_updated']:
                    (Path(directory) / f'model_{view}{suffix}.pth').write_bytes(b'fixture')
            with patch('predict.load_model', side_effect=lambda *a, **k: torch.nn.Identity()) as loader:
                predictor = DamagePredictor(directory, device=torch.device('cpu'))
            self.assertEqual('updated', predictor.model_variant)
            self.assertTrue(all('_updated.pth' in call.args[0] for call in loader.call_args_list))
            self.assertEqual(2, len(predictor.checkpoints))

    def test_invalid_fusion_weights_fail_before_inference(self):
        predictor = DamagePredictor.__new__(DamagePredictor)
        predictor.device = torch.device('cpu')
        predictor.model_hongguan = predictor.model_tishijing = None
        for weights in [(-.1, 1.1), (float('nan'), .7), (float('inf'), .7)]:
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, 'finite|non-negative'):
                predictor.predict_damage_type(weight_hongguan=weights[0], weight_tishijing=weights[1])

    def test_missing_supplied_view_does_not_silently_disappear(self):
        predictor = DamagePredictor.__new__(DamagePredictor)
        predictor.device = torch.device('cpu')
        predictor.model_hongguan = None
        predictor.model_tishijing = None
        with self.assertRaises(FileNotFoundError):
            predictor.predict_damage_type(image_hongguan_path='missing.jpg')

    def test_best_weights_are_independent_snapshot(self):
        model = torch.nn.Linear(1, 1)
        stopping = early_stopping_class()(patience=1)
        expected = model.weight.detach().clone()
        stopping(1., model)
        with torch.no_grad():
            model.weight.add_(100.)
        self.assertTrue(stopping(2., model))
        torch.testing.assert_close(expected, model.weight)

    def test_frozen_encoder_keeps_batchnorm_statistics_fixed(self):
        module = lifetime_model_module()
        config = {'model': {'embedding_dim': 16, 'dropout': 0., 'num_heads': 2,
                            'num_layers': 1, 'hidden_dim': 32}}
        model = module.create_lifetime_model(config, pretrained_backbone=False)
        model.train()
        bn = next(x for x in model.image_encoder.modules() if isinstance(x, torch.nn.BatchNorm2d))
        before = bn.running_mean.clone()
        output = model(torch.randn(2, 3, 32, 32), torch.randn(2, 7), torch.randn(2, 2), torch.randn(2, 8))
        output.sum().backward()
        torch.testing.assert_close(before, bn.running_mean)
        self.assertIsNotNone(model.image_projection.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.image_encoder.parameters()))

    def test_training_refuses_using_test_for_checkpoint_selection(self):
        import train
        model = torch.nn.Linear(1, 2)
        with self.assertRaisesRegex(ValueError, 'val'):
            train.train_model(model, {'train': [], 'test': []}, {'train': 0, 'test': 0},
                              torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=.01),
                              None, num_epochs=1, device=torch.device('cpu'), photo_type='hongguan')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

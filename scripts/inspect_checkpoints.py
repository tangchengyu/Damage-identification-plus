"""Inspect local classifier revisions without changing or training checkpoints."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def inspect_checkpoints():
    records = []
    folder = ROOT / 'sunshangshibie_tiaozhan/output/models'
    for view in ('hongguan', 'tishijing'):
        files = [folder / f'model_{view}{suffix}.pth' for suffix in ('', '_updated')]
        states = [torch.load(path, map_location='cpu', weights_only=True) for path in files]
        if states[0].keys() != states[1].keys():
            raise ValueError(f'Checkpoint architectures differ: {view}')
        changed = [key for key in states[0] if not torch.equal(states[0][key], states[1][key])]
        records.append({'view': view,
                        'files': [{'file': str(path.relative_to(ROOT)).replace('\\', '/'),
                                   'bytes': path.stat().st_size,
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files],
                        'tensor_count': len(states[0]), 'changed_tensor_count': len(changed),
                        'classifier_head_changed': any(key.startswith('backbone.fc.') for key in changed)})
    return {'scope': 'checkpoint_identity_and_tensor_difference_not_accuracy_evaluation',
            'default_variant': 'updated', 'checkpoints': records}


if __name__ == '__main__':
    print(json.dumps(inspect_checkpoints(), ensure_ascii=False, indent=2))

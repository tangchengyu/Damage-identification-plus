"""Regression checks for ambiguous historical image references."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from scripts.audit_dataset import audit


class AuditReferenceTests(unittest.TestCase):
    def test_missing_test_reference_does_not_reuse_training_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = root
            data = module / 'data'
            image = data / 'yashang/hongguan/train/4884-3.jpg'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(image)
            records = {'yashang': {
                'existing': {'path': 'data_yashang_hongguan_train_4884-3',
                             'category-vector': [0, 0, 0, 1], 'shouming': 100},
                'missing': {'path': 'data_yashang_hongguan_test_4884-3',
                            'category-vector': [0, 0, 0, 1], 'shouming': 200}}}
            metadata = data / 'DATA.json'
            metadata.write_text(json.dumps(records), encoding='utf-8')
            original_bytes = metadata.read_bytes()
            extra = module / 'extra_data'
            extra.mkdir()
            (extra / 'DATA.json').write_text('{}', encoding='utf-8')
            images, summary = audit(root)
            self.assertEqual(['existing'], [x['sample_id'] for x in images[0]['metadata_records']])
            self.assertEqual('missing', summary['datasets']['data']['unresolved_metadata'][0]['id'])
            self.assertEqual([], summary['datasets']['data']['duplicate_references'])
            self.assertEqual(original_bytes, metadata.read_bytes())


if __name__ == '__main__':
    unittest.main()

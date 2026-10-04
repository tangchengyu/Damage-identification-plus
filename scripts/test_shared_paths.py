"""Check shared data paths from several launch directories without training."""
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_function(relative_path, name):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'__file__': str(path), 'Path': Path, 'os': os, 'json': json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class SharedPathTests(unittest.TestCase):
    def test_configuration_and_pipeline_do_not_depend_on_launch_directory(self):
        load_config = load_function('shoumingyuce/main.py', 'load_config')
        pipeline_config = load_function('sunshangshibie_tiaozhan/scripts/run_pipeline.py', 'create_default_config')
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                for directory in [ROOT, ROOT / 'shoumingyuce', ROOT / 'sunshangshibie_tiaozhan', Path(temporary)]:
                    os.chdir(directory)
                    config = load_config()
                    self.assertEqual(ROOT / 'data', Path(config['data']['data_path']))
                    self.assertTrue(Path(config['data']['damage_model_path']).is_file())
                    self.assertEqual(ROOT / 'data', Path(pipeline_config()['data_dir']))
            finally:
                os.chdir(previous)

    def test_both_historical_loaders_read_existing_shared_images(self):
        code = '''
import importlib.util, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
module = sys.argv[2]
spec = importlib.util.spec_from_file_location('checked_loader', root / module / 'src/data_loader.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
function = loader.load_dataset_with_metadata if module == 'sunshangshibie_tiaozhan' else loader.load_lifetime_dataset
rows = function(str(root / 'data'))
assert len(rows) == (76 if module == 'sunshangshibie_tiaozhan' else 38), len(rows)
assert all(Path(x['image_path']).is_file() and Path(x['image_path']).is_relative_to(root / 'data') for x in rows)
print(json.dumps({'module':module, 'rows':len(rows)}))
'''
        with tempfile.TemporaryDirectory() as temporary:
            for module in ['sunshangshibie_tiaozhan', 'shoumingyuce']:
                result = subprocess.run([sys.executable, '-c', code, str(ROOT), module], cwd=temporary,
                                        capture_output=True, text=True, encoding='utf-8', timeout=60)
                self.assertEqual(0, result.returncode, result.stderr)


if __name__ == '__main__':
    unittest.main()

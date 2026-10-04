"""Desktop smoke regressions; requires a local Tk display."""
import os
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sunshangshibie_tiaozhan/src'))


class DesktopRegressionTests(unittest.TestCase):
    def test_partial_calibration_uses_the_calibrated_view(self):
        constructor = tk.Tk
        def hidden_root(*args, **kwargs):
            root = constructor(*args, **kwargs)
            root.withdraw()
            return root
        namespace = None
        with patch('tkinter.Tk', side_effect=hidden_root):
            try:
                namespace = runpy.run_path(str(ROOT / 'sunshangshibie_tiaozhan/utils.py'), run_name='partial_calibration')
                scope = namespace['measure_damage_parameters'].__globals__
                rgb = np.zeros((8, 8, 3), dtype=np.uint8)
                gray = np.zeros((8, 8), dtype=np.uint8)
                scope.update(original_image_hongguan=rgb, gray_image_hongguan=gray,
                             binary_image_hongguan=gray, pixels_per_unit_scale_hongguan=1.,
                             original_image_tishijing=rgb, gray_image_tishijing=gray,
                             binary_image_tishijing=gray, pixels_per_unit_scale_tishijing=None)
                namespace['switch_to_image_type']('tishijing')
                with patch('matplotlib.pyplot.show'):
                    text = namespace['measure_damage_parameters']('剥落')
                __import__('matplotlib.pyplot', fromlist=['close']).close('all')
                self.assertEqual('hongguan', scope['current_image_type'])
                self.assertIn('宏观', text)
            finally:
                if namespace:
                    namespace['root'].destroy()

    def test_classification_without_calibration_and_visible_model_status(self):
        constructor = tk.Tk
        def hidden_root(*args, **kwargs):
            root = constructor(*args, **kwargs)
            root.withdraw()
            return root
        namespace = None
        with patch('tkinter.Tk', side_effect=hidden_root):
            try:
                namespace = runpy.run_path(str(ROOT / 'sunshangshibie_tiaozhan/utils.py'), run_name='gui_test')
                scope = namespace['auto_detect'].__globals__
                result = {'predicted_class_zh': '剥落', 'confidence': .8, 'model_variant': 'updated',
                          'checkpoints': {}, 'weights': {'hongguan': 1., 'tishijing': 0.}}
                class Predictor:
                    def predict_damage_type(self, **kwargs):
                        return result
                captured = []
                scope.update(predictor=Predictor(), original_image_hongguan=np.zeros((8, 8, 3)),
                             img_path_hongguan='fixture.jpg',
                             show_detailed_results=lambda text, data: captured.append((text, data)),
                             measure_damage_parameters=lambda kind: self.fail('Uncalibrated measurement'))
                with patch('tkinter.messagebox.showwarning') as warning:
                    namespace['auto_detect']()
                    self.assertEqual(1, len(captured))
                    warning.assert_not_called()
                self.assertIn('updated', namespace['model_status'].get())
            finally:
                if namespace:
                    namespace['root'].destroy()


if __name__ == '__main__':
    unittest.main()

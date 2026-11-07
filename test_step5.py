"""Figure input identity, safe replacement and plotting tests."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import json
import step1,step3,step5

class FigureTests(unittest.TestCase):
    def test_import_does_not_generate_figures(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('step5_import_test',step5.__file__)
        with patch.object(Path,'mkdir',side_effect=AssertionError('Import side effect')):
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.assertTrue(callable(mod.run))
    def test_unrelated_output_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'unrelated.pdf';p.write_bytes(b'keep')
            step3.prepare_output(td,'step5_',False);self.assertEqual(p.read_bytes(),b'keep')
    def test_failed_or_old_inputs_rejected_before_plotting(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r3=root/'results3';r3.mkdir()
            (r3/'step3_run_metadata.json').write_text(json.dumps(dict(status='accepted_stage3_backtests',outputs=[])))
            with self.assertRaises(ValueError):step5.run(r3,root/'results4',root/'results5')
    def test_different_stage3_and_stage4_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r3=root/'r3';r4=root/'r4';r3.mkdir();r4.mkdir()
            (r3/'step3_run_metadata.json').write_text(json.dumps(dict(status='accepted_modelA_stage3',pipeline_model='ModelA',outputs=[])))
            (r4/'step4_run_metadata.json').write_text(json.dumps(dict(status='accepted_modelA_stage4',pipeline_model='ModelA',pipeline_revision='review_completion_v1',outputs=[],source_stage3_metadata_sha256='wrong')))
            with self.assertRaisesRegex(ValueError,'provenance'):step5.run(r3,r4,root/'r5')
    def test_changed_table_hash_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.csv';p.write_text('x\n1\n');step3.finish_metadata(td,'meta.json',dict(status='ok'),[p]);p.write_text('x\n2\n')
            with self.assertRaises(ValueError):step3.checked_manifest(td,'meta.json','ok')

if __name__=='__main__':unittest.main()

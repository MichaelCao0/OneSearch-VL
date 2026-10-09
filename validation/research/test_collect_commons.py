import pathlib
import tempfile
import unittest
from research.av_data import collect_commons as c


class CollectCommonsTest(unittest.TestCase):
    def test_html_sanitizer_keeps_words_not_markup(self):
        self.assertEqual(c.public_text('<p>A &amp; B<br>field recording</p><script>evil()</script>'), 'A & B field recording')

    def test_original_url_removes_tracking_only(self):
        self.assertEqual(c.original_url('https://upload.wikimedia.org/x/a.webm?utm_source=z'), 'https://upload.wikimedia.org/x/a.webm')

    def test_candidate_selection_is_unique_small_and_ranked(self):
        rows = {'2': {'pageid': 2, 'title':'File:B.webm','index':2,'imageinfo':[{'size':8}]}, '1': {'pageid':1,'title':'File:A.webm','index':1,'imageinfo':[{'size':5}]}, '3': {'pageid':3,'title':'File:C.webm','index':3,'imageinfo':[{'size':101}]}}
        chosen, refused = c.select_candidates(rows, set(), 100, 2)
        self.assertEqual([r['pageid'] for r in chosen], [1,2])
        self.assertEqual([r['pageid'] for r in refused], [3])
        chosen, _ = c.select_candidates(rows, {1,2}, 100, 2)
        self.assertEqual(chosen, [])

    def test_cap_accounts_failed_download_bytes(self):
        budget = c.AcquisitionBudget(max_downloads=2,max_bytes=10)
        budget.begin(8)
        budget.account(4)
        self.assertEqual(budget.bytes_received,4)
        with self.assertRaises(ValueError): budget.begin(8)
        budget.begin(5)
        with self.assertRaises(ValueError): budget.begin(1)

    def test_write_is_create_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d)/'a.json'
            c.write_json_once(path, {'x':1})
            with self.assertRaises(FileExistsError): c.write_json_once(path, {'x':2})

    def test_audio_requires_actual_non_silent_signal(self):
        self.assertFalse(c.usable_av({'streams':[{'codec_type':'video'}]}, -20))
        self.assertFalse(c.usable_av({'streams':[{'codec_type':'video'},{'codec_type':'audio'}]}, float('-inf')))
        self.assertTrue(c.usable_av({'streams':[{'codec_type':'video'},{'codec_type':'audio'}]}, -40))


if __name__ == '__main__': unittest.main()

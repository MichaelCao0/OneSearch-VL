import importlib.util,unittest
class ClueReviewRenderTests(unittest.TestCase):
 def test_embedded_publisher_text_cannot_close_page_script(self):
  self.assertIsNotNone(importlib.util.find_spec('research.av_data.clue_review'))
  from research.av_data.clue_review import render_page
  page=render_page({'cases':[{'publisher':{'description':'</script><script>alert(1)</script>'}}]})
  self.assertNotIn('</script><script>alert(1)</script>',page)
  self.assertIn('video',page);self.assertIn('下载审核记录',page)

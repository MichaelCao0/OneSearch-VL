import importlib.util,unittest
class ClueReviewRenderTests(unittest.TestCase):
 def test_embedded_publisher_text_cannot_close_page_script(self):
  self.assertIsNotNone(importlib.util.find_spec('research.av_data.clue_review'))
  from research.av_data.clue_review import render_page
  page=render_page({'cases':[{'publisher':{'description':'</script><script>alert(1)</script>'}}]})
  self.assertNotIn('</script><script>alert(1)</script>',page)
  self.assertIn('video',page);self.assertIn('下载审核记录',page)

class ActualReviewPlayerTests(unittest.TestCase):
 def test_train_count_and_compatible_default_player(self):
  from research.av_data.clue_review import render_page
  page=render_page({'summary':{'split':'train','parent_recordings':9},'cases':[{}]*21})
  self.assertIn('21 个 训练 候选，9 个录制来源',page)
  self.assertIn("format='mp4'",page);self.assertIn("v.preload='metadata'",page)
  self.assertIn('${payload.cases.length}',page);self.assertIn('播放完整视频',page)

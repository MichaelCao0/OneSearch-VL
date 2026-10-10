import unittest
from research.av_agent.native_clock_gate import check_native_clip_clocks, ClockAlignmentUnsupportedError

class NativeClockGateTests(unittest.TestCase):
    def bound(self,audio=0.0,video=0.0):
        return {'mapping':{'streams':[{'codec_type':'audio','output_start_time_s':audio},{'codec_type':'video','output_start_time_s':video}]}}
    def test_preserved_delayed_audio_rejected_before_array_consumption(self):
        with self.assertRaises(ClockAlignmentUnsupportedError):check_native_clip_clocks(self.bound(audio=1.012))
    def test_shared_nonzero_origin_cannot_be_silently_rebased(self):
        with self.assertRaises(ClockAlignmentUnsupportedError):check_native_clip_clocks(self.bound(audio=0.04,video=0.04))
    def test_container_rounding_allowance_is_bounded(self):
        check_native_clip_clocks(self.bound(video=0.001))
        with self.assertRaises(ClockAlignmentUnsupportedError):check_native_clip_clocks(self.bound(video=0.0021))
    def test_unknown_or_invalid_origins_fail_closed(self):
        for value in [None,True,float('nan'),float('inf'),10**1000]:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ClockAlignmentUnsupportedError):check_native_clip_clocks(self.bound(audio=value))
if __name__=='__main__':unittest.main()

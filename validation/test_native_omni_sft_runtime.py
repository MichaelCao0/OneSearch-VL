"""Original trainer CPU multimodal/label contracts; no model or GPU allocated."""
from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from types import MethodType,SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'research/runs/av-native-smoke-20261010-v1/runtime-setup/model-manifest.json'
PINNED=importlib.metadata.version('transformers')=='4.57.6' and MANIFEST.is_file()


@unittest.skipUnless(PINNED,'requires original trainer and pinned native Omni runtime')
class OriginalOmniTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        sys.path.insert(0,str(ROOT/'SFT/src'))

    def test_lora_targets_freeze_audio_visual_and_projector_modules(self):
        import torch
        from llamafactory.hparams import FinetuningArguments
        from llamafactory.model.model_utils.visual import patch_target_modules
        class TinyNamesOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config=SimpleNamespace(model_type='qwen2_5_omni_thinker')
                self.audio_tower=torch.nn.ModuleDict({'q_proj':torch.nn.Linear(2,2),'proj':torch.nn.Linear(2,2)})
                self.visual=torch.nn.ModuleDict({'blocks':torch.nn.ModuleDict({'q_proj':torch.nn.Linear(2,2)}),
                                                 'merger':torch.nn.ModuleDict({'q_proj':torch.nn.Linear(2,2)})})
                self.model=torch.nn.ModuleDict({'q_proj':torch.nn.Linear(2,2),'gate_proj':torch.nn.Linear(2,2)})
        args=FinetuningArguments(finetuning_type='lora',lora_target='q_proj,gate_proj',freeze_vision_tower=True,
                                freeze_multi_modal_projector=True)
        targets=patch_target_modules(TinyNamesOnly(),args,['q_proj','gate_proj'])
        self.assertEqual(set(targets),{'model.q_proj','model.gate_proj'})

    def test_finite_mkv_samples_late_marker_with_effective_pixel_budget(self):
        import numpy as np
        from research.av_train.native_sft import native_components
        from llamafactory.data.mm_plugin import _open_video_container, Qwen2VLPlugin
        snapshot=json.loads(MANIFEST.read_text())["snapshot_path"]
        _,template,_,loaded=native_components(snapshot,32768,True)
        plugin=template.mm_plugin
        with tempfile.TemporaryDirectory() as root:
            video=Path(root)/"late-marker.mkv"
            subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-f","lavfi","-i",
                "color=c=blue:size=112x56:rate=20:duration=12","-vf",
                "drawbox=x=0:y=0:w=iw:h=ih:color=red:t=fill:enable='gte(t,9)'",
                "-c:v","libx264",str(video)],check=True)
            with _open_video_container(str(video)) as (container,stream):
                self.assertEqual(stream.frames,0);self.assertIsNone(stream.duration)
                self.assertAlmostEqual(container.duration/1e6,12,places=2)
            data=plugin._regularize_videos([str(video)],image_min_pixels=3136,image_max_pixels=6272,
                                          video_fps=2,video_maxlen=24)
            self.assertEqual(len(data["videos"][0]),24)
            last=np.asarray(data["videos"][0][-1]);first=np.asarray(data["videos"][0][0])
            self.assertGreater(float(last[:,:,0].mean()),180)
            self.assertGreater(float(first[:,:,2].mean()),180)
            receipt=data["sampling_receipts"][0]
            self.assertAlmostEqual(data["durations"][0],12,places=2)
            self.assertAlmostEqual(receipt["sampled_pts_s"][-1],11.95,places=2)
            self.assertEqual(receipt["selected_frame_indices"][-1],239)
            self.assertLessEqual(receipt["max_retained_selected_frames"],24)
            # Exercise a clipped temporal budget rather than just fps sampling.
            clipped=plugin._regularize_videos([str(video)],image_min_pixels=3136,image_max_pixels=6272,
                                             video_fps=2,video_maxlen=8)
            self.assertEqual(len(clipped["videos"][0]),8)
            self.assertEqual(clipped["sampling_receipts"][0]["selected_frame_indices"][-1],239)
            self.assertAlmostEqual(clipped["fps_per_video"][0],8/12)
            loaded["processor"].video_min_pixels=3136;loaded["processor"].video_max_pixels=6272
            loaded["processor"].video_maxlen=24
            mm=plugin._get_mm_inputs([],[str(video)],[],loaded["processor"])
            _,h,w=mm["video_grid_thw"][0].tolist()
            pixels=h*w*loaded["processor"].video_processor.patch_size**2
            self.assertGreaterEqual(pixels,3136);self.assertLessEqual(pixels,6272)
            self.assertAlmostEqual(mm["video_second_per_grid"].item(),1,places=3)
            # Parent plugin keeps its existing index sampler; Omni alone overrides it.
            self.assertIsNot(type(plugin)._regularize_videos,Qwen2VLPlugin._regularize_videos)

    def test_image_and_video_processor_do_not_replace_configured_resize_bounds(self):
        from PIL import Image
        from research.av_train.native_sft import native_components
        snapshot=json.loads(MANIFEST.read_text())["snapshot_path"]
        _,template,_,loaded=native_components(snapshot,32768,True)
        native=loaded["processor"]
        native.image_min_pixels=3136;native.image_max_pixels=6272
        native.video_min_pixels=3136;native.video_max_pixels=6272
        for image in (Image.new("RGB",(14,14)),Image.new("RGB",(840,420))):
            mm=template.mm_plugin._get_mm_inputs([image],[[image,image]],[],native)
            for name in ("image_grid_thw","video_grid_thw"):
                _,h,w=mm[name][0].tolist()
                pixels=h*w*native.image_processor.patch_size**2
                self.assertGreaterEqual(pixels,3136);self.assertLessEqual(pixels,6272)

    def test_vfr_nonzero_pts_is_sampled_across_the_clip_with_bounded_frames(self):
        from research.av_train.native_sft import native_components
        snapshot=json.loads(MANIFEST.read_text())["snapshot_path"]
        _,template,_,_=native_components(snapshot,32768,True)
        with tempfile.TemporaryDirectory() as root:
            video=Path(root)/"vfr.mkv"
            subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-f","lavfi","-i",
                "testsrc2=size=56x56:rate=20:duration=5","-vf",
                "select='if(lt(t,2),not(mod(n,4)),not(mod(n,2)))',setpts=PTS+3/TB",
                "-fps_mode","vfr","-c:v","libx264",str(video)],check=True)
            data=template.mm_plugin._regularize_videos([str(video)],image_min_pixels=3136,image_max_pixels=6272,
                                                      video_fps=2,video_maxlen=8)
            receipt=data["sampling_receipts"][0]
            self.assertGreater(receipt["first_decoded_pts_s"],2.9)
            self.assertGreater(receipt["max_decoded_pts_step_s"],receipt["min_decoded_pts_step_s"]*1.5)
            self.assertEqual(len(data["videos"][0]),8)
            self.assertEqual(receipt["selected_frame_indices"][0],0)
            self.assertEqual(receipt["selected_frame_indices"][-1],receipt["decoded_frame_count"]-1)
            self.assertEqual(receipt["sampling_mode"],"uniform_decoded_pts")
            self.assertGreater(receipt["sampled_pts_s"][-1]-receipt["sampled_pts_s"][0],4.7)
            self.assertLessEqual(receipt["max_retained_selected_frames"],8)

    def test_original_supervised_processor_and_collator_mask_media_and_train_all_actions(self):
        import torch
        from research.av_train.native_sft import native_components
        from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
        from llamafactory.extras.constants import IGNORE_INDEX
        from transformers import Qwen2_5OmniConfig,Qwen2_5OmniThinkerForConditionalGeneration as Thinker
        snapshot=json.loads(MANIFEST.read_text())['snapshot_path']
        converter,template,processor,loaded=native_components(snapshot,32768,True)
        # This is test-only content, never exported or represented as a generated trajectory.
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);videos=[];audios=[]
            for i,seconds in enumerate((12,5)):
                video=root/f'test{i}.mkv';audio=root/f'test{i}.wav'
                subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-f','lavfi','-i',
                    f'testsrc2=size=56x56:rate=7:duration={seconds}','-f','lavfi','-i',
                    f'sine=frequency=440:sample_rate=16000:duration={seconds}','-c:v','libx264','-c:a','aac','-shortest',str(video)],check=True)
                subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(video),'-vn','-ac','1','-ar','16000',str(audio)],check=True)
                videos.append(str(video));audios.append(str(audio))
            action='<think>UNIT TEST ONLY</think><tool_call>{"name":"inspect_av","arguments":{"video":"video_1","start_s":1,"end_s":2}}</tool_call>'
            final='<response>UNIT TEST ONLY FINAL</response>'
            row={'system':'UNIT TEST ONLY, no model generated this sample.','tools':'','images':[],'videos':videos,'audios':audios,
                 'conversations':[{'from':'human','value':'<video><audio>UNIT TEST question'},
                                  {'from':'gpt','value':action},
                                  {'from':'human','value':'<observation>UNIT TEST observation</observation>\n<video><audio>'},
                                  {'from':'gpt','value':final}]}
            aligned=converter(row);media=[aligned[k] or [] for k in ('_images','_videos','_audios')]
            expanded=template.mm_plugin.process_messages(aligned['_prompt']+aligned['_response'],*media,loaded['processor'])
            pairs=template.encode_multiturn(loaded['tokenizer'],expanded,aligned['_system'],aligned['_tools'])
            ids,labels=processor._encode_data_example(aligned['_prompt'],aligned['_response'],aligned['_system'],aligned['_tools'],*media)
            offset=0
            self.assertEqual(len(pairs),2)
            for source,target in pairs:
                self.assertEqual(labels[offset:offset+len(source)],[IGNORE_INDEX]*len(source))
                offset+=len(source)
                self.assertEqual(labels[offset:offset+len(target)],target);offset+=len(target)
            config=Qwen2_5OmniConfig.from_pretrained(snapshot,local_files_only=True).thinker_config
            rope=SimpleNamespace(config=config,spatial_merge_size=config.vision_config.spatial_merge_size)
            for name in ('get_llm_pos_ids_for_vision','get_chunked_index','get_rope_index'):
                setattr(rope,name,MethodType(getattr(Thinker,name),rope))
            collator=MultiModalDataCollatorForSeq2Seq(tokenizer=loaded['tokenizer'],model=rope,template=template,
                                                    processor=loaded['processor'],padding=True,label_pad_token_id=IGNORE_INDEX)
            batch=collator([{'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels,
                              'images':media[0],'videos':media[1],'audios':media[2]}])
            self.assertEqual(batch['input_features'].shape[0],2)
            self.assertEqual(batch['video_grid_thw'].shape[0],2)
            self.assertEqual(batch['input_ids'].dtype,torch.int64)
            self.assertEqual(batch['position_ids'].shape[0],4)
            self.assertTrue(all(v.device.type=='cpu' for v in batch.values() if isinstance(v,torch.Tensor)))
            targets=loaded['tokenizer'].decode([x for x in labels if x!=IGNORE_INDEX],skip_special_tokens=False)
            self.assertIn(action,targets);self.assertIn(final,targets)
            self.assertNotIn('UNIT TEST observation',targets);self.assertNotIn('UNIT TEST question',targets)
            for token in (loaded['processor'].audio_token,loaded['processor'].video_token):
                token_id=loaded['tokenizer'].convert_tokens_to_ids(token)
                for value,label in zip(ids,labels):
                    if value==token_id:self.assertEqual(label,IGNORE_INDEX)


if __name__=='__main__':unittest.main()

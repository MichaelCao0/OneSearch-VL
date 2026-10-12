"""Native SFT export contracts. Fixture text is unit-test data, never a teacher."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research.av_train.native_sft import (
    answer_only_row,dataset_registration,digest,export_episodes,file_sha,
    materialize_configs,snapshot_row,validate_trajectory,
)


class NativeSftTests(unittest.TestCase):
    def fixture(self,root,split="train"):
        media=Path(root)/"query.mp4";media.write_bytes(b"UNIT TEST MEDIA ONLY")
        audio=Path(root)/"audio.wav";audio.write_bytes(b"UNIT TEST AUDIO ONLY")
        initial={"role":"user","content":[{"type":"video_url","video_url":{"url":str(media)},"native_av":True,"has_audio":True},
                                            {"type":"text","text":"UNIT TEST question"}]}
        action='<think>TEST ONLY</think><tool_call>{"name":"inspect_av","arguments":{"video":"video_1","start_s":0,"end_s":1}}</tool_call>'
        final='<think>TEST ONLY</think><response>UNIT TEST answer</response>'
        history=[{"role":"system","content":"UNIT TEST system, not generated teacher data"},initial]
        turns=[]
        for index,text in enumerate((action,final)):
            history=history+[{"role":"assistant","content":text}]
            turns.append({"turn":index,"chat_completions":deepcopy(history),"response_text":text,
                          "response":{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":text}]}}],
                                      "raw_response":{"metrics":{"model_id":"Qwen/Qwen2.5-Omni-7B","generated_tokens":10}}}})
            if index==0:
                turns[-1]["tool_output"]="<observation>UNIT TEST observation</observation>"
                history=history+[{"role":"user","content":"<tool_response>\n"+turns[-1]["tool_output"]+"\n</tool_response>"}]
        trace={"case_id":"unit-train-001","modality":"audio_video","input_mode":"native_audio_video","status":"ok",
               "question":"UNIT TEST question","input_media":{"path":str(media)},"turns":turns,
               "final_response_text":"<response>UNIT TEST answer</response>"}
        batch=Path(root)/"batch";batch.mkdir()
        (batch/"scope.json").write_text(json.dumps({"split":split,"source_ids":["clue"]}))
        (batch/"tasks-public.jsonl").write_text(json.dumps({"task_id":"unit-train-001","split":split,
                                                         "question":"UNIT TEST question","initial_sources":["clue"]})+'\n')
        (batch/"clue-bindings.private.json").write_text(json.dumps([{"clue_id":"clue","source":{"asset_path":str(media),"asset_sha256":file_sha(media)}}]))
        episodes=Path(root)/"episodes";case=episodes/"unit-train-001";case.mkdir(parents=True)
        path=case/"unit-train-001_trajectory.json";path.write_text(json.dumps(trace))
        (episodes/"inputs.private.json").write_text(json.dumps({"batch_dir":str(batch),"cases":[{"case_id":"unit-train-001",
              "question":"UNIT TEST question","source_path":str(media),"source_sha256":file_sha(media)}]}))
        verdict={"unit-train-001":{"success":True,"split":"train","reviewer_id":"unit-test-only",
                                  "trajectory_sha256":file_sha(path),"quality_tier":"silver"}}
        resolver=lambda path:(str(audio),{"operation":"UNIT TEST ONLY"})
        return trace,episodes,verdict,resolver

    def test_actual_snapshot_actions_final_and_observation_are_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            trace,_,_,resolver=self.fixture(root)
            snapshot=validate_trajectory(trace)
            row,presence,_=snapshot_row(snapshot,resolver)
            self.assertEqual([m["from"] for m in row["conversations"]],["human","gpt","human","gpt"])
            self.assertEqual(row["conversations"][1]["value"],trace["turns"][0]["response_text"])
            self.assertIn("<observation>",row["conversations"][2]["value"])
            self.assertIn("<video><audio>",row["conversations"][0]["value"])
            self.assertEqual(presence,[True]);self.assertEqual(len(row["audios"]),1)
            answer,_,_=answer_only_row(snapshot,resolver)
            self.assertEqual(answer["videos"],row["videos"])
            self.assertEqual(answer["audios"],row["audios"])
            self.assertEqual(answer["conversations"][0],row["conversations"][0])
            self.assertEqual(answer["conversations"][1]["value"],trace["final_response_text"])
            self.assertNotIn("<tool_call>",answer["conversations"][1]["value"])

    def test_silent_video_has_no_fabricated_audio(self):
        with tempfile.TemporaryDirectory() as root:
            trace,_,_,_=self.fixture(root)
            row,presence,_=snapshot_row(validate_trajectory(trace),lambda path:(None,None))
            self.assertEqual(row["audios"],[]);self.assertEqual(presence,[False])
            self.assertIn("<video>",row["conversations"][0]["value"])
            self.assertNotIn("<audio>",row["conversations"][0]["value"])

    def test_corrupted_history_or_replaced_final_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            trace,_,_,_=self.fixture(root)
            for mutation in ("history","generation","final","receipt"):
                value=deepcopy(trace)
                if mutation=="history":value["turns"][1]["chat_completions"][1]["content"][1]["text"]="rewritten"
                elif mutation=="generation":value["turns"][0]["response_text"]="invented tool"
                elif mutation=="final":value["final_response_text"]="gold answer replacement"
                else:value["turns"][0]["response"]["raw_response"]["metrics"]={}
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_trajectory(value)

    def test_train_export_requires_successful_hash_bound_verdict(self):
        with tempfile.TemporaryDirectory() as root:
            _,episodes,verdict,resolver=self.fixture(root)
            with self.assertRaisesRegex(ValueError,"verdicts required"):
                export_episodes(episodes,Path(root)/"absent")
            with patch("research.av_train.native_sft.audio_sidecar",side_effect=lambda path,*a:resolver(path)):
                result=export_episodes(episodes,Path(root)/"good",verdicts=verdict)
            self.assertEqual(result["retained_count"],1)
            self.assertFalse(result["audit_only"])
            self.assertTrue(result["use_audio_in_video"])
            self.assertTrue((Path(root)/"good/dataset_info.json").is_file())
            invalid=deepcopy(verdict);invalid["unit-train-001"]["trajectory_sha256"]="0"*64
            result=export_episodes(episodes,Path(root)/"bad",verdicts=invalid)
            self.assertEqual(result["retained_count"],0)
            self.assertEqual(result["rejected_count"],1)

    def test_dev_split_cannot_export_train_or_materialize_launch_config(self):
        with tempfile.TemporaryDirectory() as root:
            _,episodes,verdict,resolver=self.fixture(root,split="dev")
            with self.assertRaisesRegex(ValueError,"audit-only"):
                export_episodes(episodes,Path(root)/"train",verdicts=verdict)
            out=Path(root)/"audit"
            with patch("research.av_train.native_sft.audio_sidecar",side_effect=lambda path,*a:resolver(path)):
                result=export_episodes(episodes,out,audit_only=True)
            self.assertTrue(result["audit_only"])
            self.assertFalse((out/"dataset_info.json").exists())
            audit=Path(root)/"report";audit.mkdir()
            (audit/"label-media-audit.private.json").write_text(json.dumps({"status":"passed","audit_only":True}))
            with self.assertRaisesRegex(ValueError,"dev audit cannot train"):
                materialize_configs(out,audit,"not-read",Path(root)/"launch")

    def test_authoritative_split_media_binding_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            _,episodes,verdict,_=self.fixture(root)
            batch=Path(root)/"batch/clue-bindings.private.json"
            value=json.loads(batch.read_text());value[0]["source"]["asset_sha256"]="0"*64;batch.write_text(json.dumps(value))
            result=export_episodes(episodes,Path(root)/"blocked",verdicts=verdict)
            self.assertEqual(result["retained_count"],0)
            self.assertIn("split-bound clue",(Path(root)/"blocked/rejections.private.json").read_text())

    def test_train_config_requires_verified_original_sampler_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            _,episodes,verdict,resolver=self.fixture(root)
            data=Path(root)/"train-data"
            with patch("research.av_train.native_sft.audio_sidecar",side_effect=lambda path,*a:resolver(path)):
                export_episodes(episodes,data,verdicts=verdict)
            audit=Path(root)/"audit";audit.mkdir()
            (audit/"label-media-audit.private.json").write_text(json.dumps({
                "status":"passed","audit_only":False,"dataset_summary_sha256":file_sha(data/"summary.json"),
                "original_sampler_native_video_coverage_verified":False}))
            with self.assertRaisesRegex(ValueError,"video sampler coverage"):
                materialize_configs(data,audit,"not-read",Path(root)/"blocked-config")

    def test_train_config_accepts_fresh_receipts_but_rejects_stale_source_or_media(self):
        import yaml
        from research.av_train.native_sft import ROOT
        with tempfile.TemporaryDirectory() as root:
            _,episodes,verdict,resolver=self.fixture(root)
            data=Path(root)/"train-data"
            with patch("research.av_train.native_sft.audio_sidecar",side_effect=lambda path,*a:resolver(path)):
                export_episodes(episodes,data,verdicts=verdict)
            audit=Path(root)/"audit";audit.mkdir()
            audit_file=audit/"label-media-audit.private.json"
            video=Path(root)/"query.mp4"
            source="data/mm_plugin.py"
            report={"status":"passed","audit_only":False,"dataset_summary_sha256":file_sha(data/"summary.json"),
                    "original_sampler_native_video_coverage_verified":True,"cutoff_len":32768,
                    "native_source_sha256":{source:file_sha(ROOT/"SFT/src/llamafactory"/source)},
                    "video_sampling_coverage_receipts":[{"coverage_verified":True,"processor_do_resize":False,
                                                        "video_path":str(video),"video_sha256":file_sha(video)}]}
            manifest=Path(root)/"manifest.json"
            manifest.write_text(json.dumps({"snapshot_path":str(Path(root)/"UNIT_TEST_ONLY_NO_MODEL"),
                                           "repository_id":"Qwen/Qwen2.5-Omni-7B","all_lfs_verified":True}))
            audit_file.write_text(json.dumps(report))
            commands=materialize_configs(data,audit,manifest,Path(root)/"fresh-config")
            self.assertEqual(len(commands),2)
            cfg=yaml.safe_load((Path(root)/"fresh-config/train-trajectory.yaml").read_text())
            self.assertEqual(cfg["template"],"qwen2_omni");self.assertTrue(cfg["use_audio_in_video"])
            self.assertFalse(cfg["train_on_prompt"]);self.assertFalse(cfg["mask_history"])
            stale=deepcopy(report);stale["native_source_sha256"][source]="0"*64;audit_file.write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError,"trainer source changed"):
                materialize_configs(data,audit,manifest,Path(root)/"stale-source")
            audit_file.write_text(json.dumps(report));video.write_bytes(b"CHANGED UNIT TEST ONLY")
            with self.assertRaisesRegex(ValueError,"audited video changed"):
                materialize_configs(data,audit,manifest,Path(root)/"stale-media")

    def test_protocol_repair_cannot_be_exported_as_executed_training_action(self):
        with tempfile.TemporaryDirectory() as root:
            trace,_,_,_=self.fixture(root)
            trace["turns"][0]["protocol_repair"]=True
            with self.assertRaisesRegex(ValueError,"protocol repair"):
                validate_trajectory(trace)

    def test_unexecuted_partial_or_mismatched_actions_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            trace,_,_,_=self.fixture(root)
            for mutation in ("missing_output", "changed_observation", "two_calls", "mixed_final", "malformed_call"):
                value=deepcopy(trace)
                if mutation=="missing_output":value["turns"][0].pop("tool_output")
                elif mutation=="changed_observation":value["turns"][0]["tool_output"]="<observation>different</observation>"
                else:
                    index=1 if mutation=="mixed_final" else 0
                    turn=value["turns"][index]
                    old=turn["response_text"]
                    action=trace["turns"][0]["response_text"]
                    new=old+action if mutation in {"two_calls","mixed_final"} else '<tool_call>{BAD JSON}</tool_call>'
                    turn["response_text"]=new
                    turn["response"]["candidates"][0]["content"]["parts"][0]["text"]=new
                    turn["chat_completions"][-1]["content"]=new
                    if index==0:value["turns"][1]["chat_completions"][2]["content"]=new
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    validate_trajectory(value)

    def test_registration_preserves_native_media_columns(self):
        registration=dataset_registration("trajectory.json")
        self.assertEqual(registration["formatting"],"sharegpt")
        for key in ("images","videos","audios"):
            self.assertEqual(registration["columns"][key],key)


if __name__=="__main__":unittest.main()

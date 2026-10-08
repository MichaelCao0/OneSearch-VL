# Enable faulthandler before libraries that modify signal-stack state.
# Ray can then reuse it without reallocating the signal stack during startup.
import faulthandler as _faulthandler
try:
    _faulthandler.enable(open("/dev/null", "w"))
except Exception:
    pass

import os

import hydra

from vision_deepresearch_async_workflow.deepresearch_tools_async_executor import (
    get_all_tools,
)
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    DeepResearchWorkflow,
)

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import deepresearch_reward_fn_async
from rllm.trainer.agent_trainer import AgentTrainer
from vision_deepresearch_async_workflow.tool_contract import validate_sft_tools


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer_megatron",
    version_base=None,
)
def main(config):
    tools = get_all_tools()
    validate_sft_tools(tools)
    _ds_name = os.environ.get("RL_DATASET_NAME", "OneSearch-VL-RL-10K")
    train_dataset = DatasetRegistry.load_dataset(_ds_name, "train")
    if train_dataset is None:
        raise FileNotFoundError(f"Register train for {_ds_name!r} with RL/register_dataset.py first.")
    # The public RL recipe is train-only. Do not reuse a stale registered test
    # split or framework example validation paths.
    config.data.val_files = []

    trainer = AgentTrainer(
        workflow_class=DeepResearchWorkflow,
        workflow_args={
            "reward_function": deepresearch_reward_fn_async,
            "tools": tools,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=None,
    )
    trainer.train()


if __name__ == "__main__":
    main()

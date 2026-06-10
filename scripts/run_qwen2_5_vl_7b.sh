set -x
ENGINE=${1:-vllm}
exp_id=qwen2_5_vl_7b_taro
output_dir=outputs/${exp_id}
mkdir -p $output_dir

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, 
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0
export VLLM_ASCEND_ENABLE_NZ=0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +trainer.warmup_epochs=1 \
    +algorithm.off_policy_mode=mask \
    +actor_rollout_ref.actor.enforce_on_policy=False \
    +custom_reward_function.reward_kwargs.temp_iou_thresh=0.7 \
    +custom_reward_function.reward_kwargs.temp_margin=0.0 \
    +custom_reward_function.reward_kwargs.temp_alpha=0.3 \
    +actor_rollout_ref.temp_diff_thresh=mean \
    custom_reward_function.path=./src/rewards.py \
    custom_reward_function.name=compute_vtg_score \
    reward_model.use_reward_loop=False \
    data.custom_cls.path=./src/vtg_dataset.py \
    data.custom_cls.name=VTGDataset \
    data.train_files=data/train.json \
    data.val_files=data/test.json \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=7168 \
    data.max_response_length=1024 \
    data.truncation='error' \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=32 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt="{image: 0, video: 1, audio: 0}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "swanlab"]' \
    trainer.project_name=taro \
    trainer.experiment_name=$exp_id \
    trainer.default_local_dir="${output_dir}/checkpoints" \
    trainer.rollout_data_dir="${output_dir}/rollout_data" \
    trainer.validation_data_dir="${output_dir}/validation_data" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=3 $@

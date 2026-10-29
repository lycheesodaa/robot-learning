"""LeRobot loaders for the v6 LoRA checkpoint and zero-shot MolmoAct2 (rulebased eval_policy.build_policy settings)."""
from pathlib import Path

import numpy as np

_POLICIES = {}


def snapshot_path(config, root):
    return Path(root)/'artifacts/models'/config['model']/config['revision']


def load_policy(config, root):
    """One resident model per process; policy.kind 'zeroshot' loads the base checkpoint, otherwise v6."""
    if config.get('kind') == 'zeroshot': return load_zeroshot(config, root)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    path = snapshot_path(config, root)
    base = Path(root)/'artifacts/models'/config['base_model']/config['base_revision']
    if not (path/'model.safetensors').exists():
        raise FileNotFoundError(f'{path}: download {config["repo_id"]}@{config["revision"]} first')
    tokenizer = Path(root)/'artifacts/models'/config['action_tokenizer']/config['action_tokenizer_revision']
    if not (base/'config.json').exists():
        raise FileNotFoundError(f'{base}: run scripts/download_models.py first')
    if not (tokenizer/'tokenizer.json').exists():
        raise FileNotFoundError(f'{tokenizer}: download allenai/MolmoAct2-FAST-Tokenizer@{config["action_tokenizer_revision"]} first')
    if config['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not visible')
    key = (str(path), config['device'])
    if key not in _POLICIES:
        _POLICIES.clear()
        # v6's config names the base by Hub id with no revision; its weights then replace the base's.
        policy_config = PreTrainedConfig.from_pretrained(str(path))
        policy_config.checkpoint_path = str(base)
        policy = MolmoAct2Policy.from_pretrained(str(path), config=policy_config)
        policy.config.inference_action_mode = config['inference_action_mode']   # not saved at train time
        policy.to(config['device'])
        # The input packer loads the base processor and the FAST tokenizer (action_mode=both) by Hub id too.
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(path), preprocessor_overrides={
            'molmoact2_pack_inputs': {'checkpoint_path': str(base), 'discrete_action_tokenizer': str(tokenizer)}})
        policy.config.per_episode_seed = config['per_episode_seed']
        policy.config.n_action_steps = config['n_action_steps']
        policy.eval()
        _POLICIES[key] = (policy, pre, post)
    return _POLICIES[key], {'model_id': config['repo_id'], 'revision': config['revision'], 'snapshot': str(path),
                            'base_snapshot': str(base),
                            'model_dtype': str(_POLICIES[key][0].config.model_dtype)}


def load_zeroshot(config, root):
    """allenai/MolmoAct2-SO100_101 as rulebased --policy zeroshot, normalized with the checkpoint's own norm_tag stats."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_policy_config, make_pre_post_processors
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    path = snapshot_path(config, root)
    tokenizer = Path(root)/'artifacts/models'/config['action_tokenizer']/config['action_tokenizer_revision']
    if not (path/'norm_stats.json').exists():
        raise FileNotFoundError(f'{path}: run scripts/download_models.py first')
    if not (tokenizer/'tokenizer.json').exists():
        raise FileNotFoundError(f'{tokenizer}: download allenai/MolmoAct2-FAST-Tokenizer@{config["action_tokenizer_revision"]} first')
    if config['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not visible')
    key = (str(path), config['device'])
    if key not in _POLICIES:
        _POLICIES.clear()
        cfg = make_policy_config(
            'molmoact2', checkpoint_path=str(path), norm_tag=config['norm_tag'], discrete_action_tokenizer=str(tokenizer),
            model_dtype=config['model_dtype'], chunk_size=config['chunk_size'], n_action_steps=config['chunk_size'],
            setup_type=config['setup_type'], control_mode=config['control_mode'],
            image_keys=['observation.images.cam0', 'observation.images.cam1'], normalize_gripper=True,
            inference_action_mode=config['inference_action_mode'], device=config['device'])
        cfg.input_features = {k: PolicyFeature(FeatureType.VISUAL, (3, 480, 640))
                              for k in ('observation.images.cam0', 'observation.images.cam1')}
        cfg.input_features['observation.state'] = PolicyFeature(FeatureType.STATE, (6,))
        cfg.output_features = {'action': PolicyFeature(FeatureType.ACTION, (6,))}
        policy = MolmoAct2Policy(cfg).to(config['device'])
        pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=None)   # None -> norm_tag stats
        policy.config.per_episode_seed = config['per_episode_seed']
        policy.config.n_action_steps = config['n_action_steps']
        policy.eval()
        _POLICIES[key] = (policy, pre, post)
    return _POLICIES[key], {'model_id': config['repo_id'], 'revision': config['revision'], 'snapshot': str(path),
                            'model_dtype': str(_POLICIES[key][0].config.model_dtype)}


def begin_episode(bundle, seed):
    """Fresh action queue and a flow-sampling generator derived from `seed` only."""
    import torch
    policy, _, _ = bundle
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    policy.config.eval_seed = seed
    policy.reset()
    policy._rollout_index_for_task = -1
    policy._rollout_task_key = None


def select_action(bundle, cam0, cam1, state, task):
    """One checkpoint-convention action; LeRobot re-plans every n_action_steps calls."""
    import torch
    policy, pre, post = bundle
    device = policy.config.device
    def image(x):
        return torch.from_numpy(x).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(device)
    batch = {'observation.images.cam0': image(cam0), 'observation.images.cam1': image(cam1),
             'observation.state': torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(device),
             'task': [task]}
    with torch.no_grad():
        return post(policy.select_action(pre(batch)))[0].detach().float().cpu().numpy().astype(np.float64)

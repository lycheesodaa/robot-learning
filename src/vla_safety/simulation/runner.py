"""Streaming, same-state closed-loop experiments and diagnostics."""
from contextlib import nullcontext
from copy import deepcopy
import html
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np

from vla_safety.concepts.archive import atomic_json, digest, provenance, RunArchive
from vla_safety.concepts.capture import capture_components, enumerate_components
from vla_safety.concepts.runner import load_policy
from vla_safety.interventions.hooks import HookSpec, intervention_hooks
from vla_safety.policies.wire import encode_action
from .environment import SO101MujocoEnvironment, validate_config, JOINTS
from .scoring import ReleaseTracker


def instruction(config,fragile,rule):
    t=config['task']
    return ' '.join([t['base_instruction'],t['fragile_instruction' if fragile else 'durable_instruction'],
                     t[rule+'_instruction'].format(limit_cm=t['max_release_height_m']*100)])


def prediction_seed(seed,index):
    return int(np.random.SeedSequence([seed,index]).generate_state(1)[0])


def _jsonline(file,value):
    file.write(json.dumps(value,allow_nan=False)+'\n'); file.flush()


def _report(folder,result):
    (folder/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Release rollout</title>'
        '<h1>Simulated release rollout</h1><video controls width="960" src="rollout.mp4"></video>'
        '<p>Top view then side view. Simulated behavior; no breakage model.</p><pre>'+
        html.escape(json.dumps({k:v for k,v in result.items() if k!='trace'},indent=2))+'</pre>')


def _seal(folder):
    atomic_json(folder/'files.json',{str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*'))
                                  if p.is_file() and p.name!='files.json'})


def target_specs(config,root,policy):
    """Read frozen rankings; do not fit/select using these evaluation rollouts."""
    settings=config['interventions']
    if not settings['ranking_archive']:
        return {'baseline':([],{'condition':'baseline'})}
    archive=Path(root)/settings['ranking_archive']
    source=archive/'analysis/fragility_height_interaction/results.json'
    ranking=json.loads(source.read_text())
    archived_provenance=json.loads((archive/'provenance.json').read_text())
    model_config=json.loads((Path(root)/'configs/models.json').read_text())['so101']
    if archived_provenance.get('model_id')!=model_config['repo_id'] or archived_provenance.get('revision')!=model_config['revision']:
        raise ValueError('Ranking archive checkpoint does not match rollout checkpoint')
    inventory={c.name:c for c in enumerate_components(policy.model)}
    eligible=[r for r in ranking['rankings'] if r['component'] in inventory]
    if not eligible: raise ValueError('Ranking has no components supported by this checkpoint')
    preferred=[r for r in eligible if inventory[r['component']].kind=='mlp_neurons']
    target=(preferred or eligible)[0]['component'];component=inventory[target]
    channels=list(dict.fromkeys(r['channel'] for r in eligible if r['component']==target))
    group=channels[:settings['group_size']]
    width=next(c['channels'] for c in ranking['components'] if c['component']==target)
    random=np.random.default_rng(settings['random_seed']).choice(width,size=len(group),replace=False).tolist()
    if not group or any(type(i) is not int or i<0 or i>=width for i in group):
        raise ValueError('Invalid ranked channel indices')
    result={'baseline':([],{'condition':'baseline'})}
    for name,neurons,strength in [('zero',group,0.),('top_neuron',group[:1],1.),('top_group',group,1.),('random_group',random,1.),('whole_component',[],1.)]:
        result[name]=([HookSpec(component.module,'ablate',strength,tuple(neurons),site=component.site)],
                      {'condition':name,'component':target,'neurons':neurons,'strength':strength,
                       'ranking_source':str(source),'ranking_sha256':digest(source),
                       'ranking_split':ranking.get('ranking_split'),'evidence':ranking.get('evidence'),
                       'interpretation':'exploratory_transfer_from_frozen_concept_archive'})
    return result


def _capture_save(folder,index,captures):
    arrays={};entries=[]
    for ci,(name,records) in enumerate(captures.items()):
        for invocation,record in enumerate(records):
            key=f'c{ci}_call{invocation}';arrays[key]=record['mean']
            entry={'component':name,'invocation':invocation,'shape':record['shape'],'mean_key':key}
            if 'raw' in record: arrays[key+'_raw']=record['raw'];entry['raw_key']=key+'_raw'
            entries.append(entry)
    path=folder/'activations';path.mkdir(exist_ok=True)
    np.savez_compressed(path/f'{index:05d}.npz',**arrays)
    atomic_json(path/f'{index:05d}.json',{'capture_phase':'before_intervention','components':entries})


def run_episode(config,root,folder,*,height='high',fragile=True,rule='explicit',seed=1000,
                layout=None,controller='molmoact2',policy=None,specs=(),descriptor=None,
                initial_snapshot=None,render=True):
    validate_config(config)
    if controller not in ('molmoact2','scripted','scripted-high'):
        raise ValueError('Unknown controller')
    if type(seed) is not int or seed<0 or type(fragile) is not bool or rule not in ('implicit','explicit'):
        raise ValueError('Invalid episode factors')
    root=Path(root);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/'config.json',config)
    descriptor=descriptor or {'condition':'baseline'}
    condition={'height':height,'fragile':fragile,'rule':rule,'seed':seed,'layout':layout or config['suite']['layouts'][0],
               'controller':controller,'intervention':descriptor,'evidence':'simulated_instruction_conditioning',
               'instruction':instruction(config,fragile,rule)}
    atomic_json(folder/'condition.json',condition)
    atomic_json(folder/'status.json',{'status':'running'})
    tracker=ReleaseTracker(config['task'],fragile)
    env=writer=None;result=None;snapshot=None
    termination='error';index=0;physics_index=0;action_index=0;last_gripper_z=None
    summary={'max_joint_displacement_rad':0.,'max_arm_joint_displacement_rad':0.,'total_object_vertical_travel_m':0.,'clipped_action_count':0}
    try:
        env=SO101MujocoEnvironment(config,root,layout)
        text=condition['instruction']
        if initial_snapshot is None: env.reset(height,text,seed)
        else: env.restore(initial_snapshot,text)
        snapshot=env.snapshot();np.savez_compressed(folder/'initial_state.npz',**snapshot)
        (folder/'scene.xml').write_text(env.xml)
        metadata=provenance(root)
        metadata.update(mujoco=env.mj.__version__,asset_revision=config['assets']['revision'],
                        asset_hashes={p.name:digest(p) for p in env.assets.glob('*.xml')},
                        mesh_hashes={p.name:digest(p) for p in (env.assets/'assets').glob('*.stl')},
                        initial_state_sha256=digest(folder/'initial_state.npz'),
                        calibration=config['calibration'],initialization=getattr(env,'initialization',{'restored':True}))
        if controller=='molmoact2':
            if policy is None:
                policy,model_metadata=load_policy(config['policy'],root)
                metadata.update(model_metadata)
            else:
                model_config=json.loads((root/'configs/models.json').read_text())['so101']
                metadata.update(model_id=model_config['repo_id'],revision=model_config['revision'])
        metadata['controller']=controller
        atomic_json(folder/'provenance.json',metadata)
        for source in [* (root/'src/vla_safety/simulation').glob('*.py'),root/'src/vla_safety/interventions/hooks.py',root/'src/vla_safety/policies/molmoact2.py',root/'scripts/concept_experiment.py']:
            target=folder/'source'/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        capture_config=config['capture']
        components=enumerate_components(policy.model,capture_config['components']) if capture_config['enabled'] and controller=='molmoact2' else []
        if specs and controller!='molmoact2': raise ValueError('Neural interventions require MolmoAct2')
        if not render and controller=='molmoact2': raise ValueError('MolmoAct2 requires rendered observations')
        frames=None
        if render:
            import imageio.v2 as imageio
            frames=env.render()
            writer=imageio.get_writer(str(folder/'rollout.mp4'),fps=config['control']['video_fps'],codec='libx264',macro_block_size=2)
        first=env.telemetry();tracker.update(first)
        initial_q=np.asarray(first['joint_radians']);last_height=first['height_m']
        duration=config['control']['duration_s'];max_steps=round(duration*config['control']['physics_hz'])
        video_every=config['control']['physics_hz']//config['control']['video_fps']
        with (folder/'telemetry.jsonl').open('w') as telemetry,(folder/'predictions.jsonl').open('w') as predictions,(folder/'executed_actions.jsonl').open('w') as actions,(folder/'video_frames.jsonl').open('w') as video_log:
            _jsonline(telemetry,first)
            if writer:
                writer.append_data(np.concatenate(frames,axis=1));_jsonline(video_log,{'frame':0,'time_s':0.})
            while physics_index<max_steps:
                if (folder/'CANCEL').exists() or (folder.parent/'CANCEL').exists():
                    termination='cancelled';break
                seed_i=prediction_seed(seed,index)
                observation=None;image_paths=[]
                if render:
                    observation=env.observe()
                    images=folder/'observations';images.mkdir(exist_ok=True)
                    for name,img in zip(('top','side'),observation.images):
                        path=images/f'{index:05d}_{name}.png';img.save(path);image_paths.append(str(path.relative_to(folder)))
                state=env.calibration.to_policy(env.data.qpos[env.qadr]).tolist()
                obs_time=float(env.data.time)
                if index==0:
                    atomic_json(folder/'scene_manifest_row.json',{
                        'id':f"{env.layout['id']}-f{int(fragile)}-{height}",'group_id':env.layout['id'],'matched_scene_id':env.layout['id'],
                        'split':env.layout['split'],'fragility_evidence':'instruction_and_assigned_metadata','height_evidence':'simulator_geometry','images':image_paths,'state':state,'embodiment':'so101',
                        'instruction':' '.join([config['task']['base_instruction'],config['task']['fragile_instruction' if fragile else 'durable_instruction']]),
                        'release_annotation':{'fragile':fragile,'height_m':first['height_m'],'holding':True,'source':'simulator geometry; fragility supplied by instruction and assigned metadata'}})
                started=time.perf_counter()
                captures={}
                if controller=='molmoact2':
                    context=capture_components(policy.model,components,raw=capture_config['raw'],max_bytes=int(capture_config['max_mib_per_prediction']*1024**2)) if components else nullcontext({})
                    # Register capture first; suppression remains scoped to every inference call.
                    with context as captures, intervention_hooks(policy.model,list(specs)):
                        chunk=policy.predict(observation,seed=seed_i,num_steps=config['policy']['num_steps'])
                    if components and any(not values for values in captures.values()):
                        raise RuntimeError('A selected capture component did not execute')
                else:
                    chunk=env.scripted_chunk(release_high=controller=='scripted-high')
                elapsed=time.perf_counter()-started
                env.validate_action(chunk)
                if components: _capture_save(folder,index,captures)
                used=min(config['control']['actions_per_prediction'],len(chunk.values))
                _jsonline(predictions,{'index':index,'time_s':obs_time,'seed':seed_i,'instruction':text,'images':image_paths,'state':state,
                                      'action':encode_action(chunk),'planned_executed_count':used,'inference_wall_s':elapsed,'intervention':descriptor})
                for native in chunk.values[:used]:
                    before=env.telemetry();proposed,executed=env.set_target(native)
                    clipped=bool(np.any(proposed!=executed));summary['clipped_action_count']+=int(clipped)
                    opening=bool(executed[5]>before['joint_radians'][5]+config['task']['opening_command_delta_rad'])
                    # Kinematic target displacement is distinct from executed velocity.
                    saved=env.snapshot_for_ik()
                    env.data.qpos[env.qadr]=executed;env.forward()
                    target_z=float(env.data.site_xpos[env.siteid,2]);env.restore_for_ik(saved)
                    _jsonline(actions,{'index':action_index,'prediction_index':index,'time_s':before['time_s'],
                                      'proposed_native':native.tolist(),'proposed_radians':proposed.tolist(),'executed_radians':executed.tolist(),
                                      'clipped':clipped,'opening_command':opening,
                                      'target_gripper_delta_z_m':target_z-before['gripper_position_m'][2]})
                    for _ in range(env.substeps):
                        row=env.physics_step();physics_index+=1
                        row.update(action_index=action_index,prediction_index=index,opening_command=opening)
                        row['gripper_velocity_z_m_s']=(row['gripper_position_m'][2]-(last_gripper_z if last_gripper_z is not None else before['gripper_position_m'][2]))*config['control']['physics_hz']
                        last_gripper_z=row['gripper_position_m'][2]
                        tracker.update(row,opening)
                        summary['max_joint_displacement_rad']=max(summary['max_joint_displacement_rad'],float(np.max(np.abs(np.asarray(row['joint_radians'])-initial_q))))
                        summary['max_arm_joint_displacement_rad']=max(summary['max_arm_joint_displacement_rad'],float(np.max(np.abs(np.asarray(row['joint_radians'])[:5]-initial_q[:5]))))
                        summary['total_object_vertical_travel_m']+=abs(row['height_m']-last_height);last_height=row['height_m']
                        row['observed_release_confirmed']=tracker.released
                        _jsonline(telemetry,row)
                        if writer and physics_index%video_every==0:
                            writer.append_data(np.concatenate(env.render(),axis=1));_jsonline(video_log,{'frame':physics_index//video_every,'time_s':row['time_s']})
                        if tracker.success or physics_index>=max_steps: break
                    action_index+=1
                    if tracker.success or physics_index>=max_steps: break
                index+=1
                if tracker.success:
                    termination='success';break
            else: termination='timeout'
            if termination=='error': termination='timeout'
        result=tracker.result(complete=termination!='cancelled')
        summary['arm_inactive']=summary['max_arm_joint_displacement_rad']<config['task']['inactivity_joint_displacement_rad']
        result.update(termination=termination,controller=controller,condition=condition,summary=summary,
                      predictions=index,executed_actions=action_index,simulation_duration_s=float(env.data.time))
    except BaseException as exc:
        result=tracker.result(complete=False) if tracker.steps else {'status':'unavailable','release_occurred':None,'first_release_height_m':None,'task_success':None}
        termination='cancelled' if isinstance(exc,KeyboardInterrupt) else 'error'
        result.update(termination=termination,error=str(exc),condition=condition,controller=controller)
        (folder/'error.txt').write_text(traceback.format_exc())
        if env is not None:
            np.savez_compressed(folder/'failure_state.npz',integration=env.snapshot_for_ik())
    finally:
        cleanup_errors=[]
        for resource in (writer,env):
            if resource is not None:
                try: resource.close()
                except Exception as exc: cleanup_errors.append(str(exc))
        if cleanup_errors and result is not None:
            result['cleanup_errors']=cleanup_errors
            result['termination']=termination='error'
        if result is not None:
            atomic_json(folder/'result.json',result)
            if 'trace' in result: atomic_json(folder/'release_trace.json',result['trace'])
            atomic_json(folder/'status.json',{'status':termination})
            _report(folder,result);_seal(folder)
    return result,snapshot


def run_suite(config,root,folder,controller='molmoact2',render=True):
    root=Path(root);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/'config.json',config)
    atomic_json(folder/'status.json',{'status':'running'})
    policy=None;specifications={'baseline':([],{'condition':'baseline'})}
    results=[];scenes={};failure=None;initializations=[]
    try:
        if controller=='molmoact2':
            policy,_=load_policy(config['policy'],root)
            specifications=target_specs(config,root,policy)
        requested=config['interventions']['conditions']
        if controller!='molmoact2': requested=['baseline']
        # Each layout/height gets exactly one validated snapshot, reused for all seeds and prompts.
        for layout,height in itertools.product(config['suite']['layouts'],config['suite']['heights']):
            env=SO101MujocoEnvironment(config,root,layout)
            try:
                env.reset(height);snapshot=env.snapshot()
                initializations.append({'layout':layout['id'],'height':height,**env.initialization})
                atomic_json(folder/'initializations.json',initializations)
            finally: env.close()
            for seed,fragile,rule,name in itertools.product(config['suite']['seeds'],config['suite']['fragilities'],config['suite']['rules'],requested):
                if (folder/'CANCEL').exists():
                    failure='cancelled';break
                stem=f"{layout['id']}-{height}-f{int(fragile)}-{rule}-s{seed}-{name}"
                specs,descriptor=specifications[name]
                print(stem,flush=True)
                result,_=run_episode(config,root,folder/stem,height=height,fragile=fragile,rule=rule,seed=seed,layout=layout,
                                     controller=controller,policy=policy,specs=specs,descriptor=descriptor,initial_snapshot=snapshot,render=render)
                results.append({'episode':stem,**{k:v for k,v in result.items() if k!='trace'}})
                atomic_json(folder/'results.json',results)
                scene_path=folder/stem/'scene_manifest_row.json'
                if name=='baseline' and scene_path.exists() and render:
                    scene=json.loads(scene_path.read_text());scene['images']=[f'{stem}/{p}' for p in scene['images']]
                    scenes.setdefault(scene['id'],scene)
                if result['termination'] in ('error','cancelled'):
                    failure=result['termination'];break
            if failure: break
    except BaseException:
        failure='error';(folder/'error.txt').write_text(traceback.format_exc())
    finally:
        (folder/'scenes.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in scenes.values()))
        # Failed episodes are excluded from behavioral rates and counted explicitly.
        groups={}
        for r in results:
            cond=r['condition'];key=f"f{int(cond['fragile'])}-{cond['height']}-{cond['rule']}-{cond['intervention']['condition']}"
            groups.setdefault(key,[]).append(r)
        summary={}
        for key,group in groups.items():
            valid=[r for r in group if r['termination'] in ('success','timeout')]
            heights=[r['first_release_height_m'] for r in valid if r.get('first_release_height_m') is not None]
            summary[key]={'episodes':len(group),'valid_episodes':len(valid),'failed_or_cancelled':len(group)-len(valid),
                          'release_rate_within_horizon':sum(bool(r['release_occurred']) for r in valid)/len(valid) if valid else None,
                          'placement_success_rate':sum(bool(r['task_success']) for r in valid)/len(valid) if valid else None,
                          'mean_release_height_m':float(np.mean(heights)) if heights else None,
                          'rule_violation_count':sum(r.get('unsafe_fragile_release') is True for r in valid),
                          'no_release_count':sum(not r['release_occurred'] for r in valid),
                          'slip_without_opening_count':sum(r.get('release_kind')=='slip_without_opening' for r in valid),
                          'inactive_arm_count':sum(r.get('summary',{}).get('arm_inactive',False) for r in valid),
                          'mean_release_time_s':float(np.mean([r['release_time_s'] for r in valid if r.get('release_time_s') is not None])) if any(r.get('release_time_s') is not None for r in valid) else None}
        atomic_json(folder/'summary.json',summary)
        atomic_json(folder/'status.json',{'status':failure or 'completed','episodes':len(results)})
        _seal(folder)
    return {'status':failure or 'completed','episodes':len(results)}


def run_checks(config,root,folder,render=True):
    """Apparatus tests, explicitly separate from policy competence."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/'config.json',config)
    checks=[];ok=True
    try:
        for layout,height in itertools.product(config['suite']['layouts'],['low','high']):
            env=SO101MujocoEnvironment(config,root,layout)
            try:
                env.reset(height);snapshot=env.snapshot()
                native=env.calibration.to_policy(env.data.qpos[env.qadr])
                roundtrip=float(np.max(np.abs(env.calibration.to_sim(native)-env.data.qpos[env.qadr])))
                for _ in range(env.substeps):env.physics_step()
                first=env.snapshot()['integration'];env.restore(snapshot)
                for _ in range(env.substeps):env.physics_step()
                replay_error=float(np.max(np.abs(first-env.snapshot()['integration'])))
                stem=f"{layout['id']}-{height}";env.restore(snapshot)
                np.savez_compressed(folder/f'{stem}-initial.npz',**snapshot)
                if render:
                    from PIL import Image
                    for name,frame in zip(('top','side'),env.render()):Image.fromarray(frame).save(folder/f'{stem}-{name}.png')
                check={'layout':layout['id'],'height':height,**env.initialization,'roundtrip_error_rad':roundtrip,'replay_error':replay_error}
                if roundtrip>1e-12 or replay_error>1e-12:raise AssertionError(check)
                checks.append(check)
                if layout==config['suite']['layouts'][0] and height=='high':
                    sweeps=[]
                    for j,name in enumerate(JOINTS):
                        env.restore(snapshot);before=env.data.qpos[env.qadr[j]]
                        targets=env.data.ctrl[env.aids].copy();targets[j]=min(before+.05,env.model.actuator_ctrlrange[env.aids[j],1])
                        env.data.ctrl[env.aids]=targets
                        for _ in range(round(.5*config['control']['physics_hz'])):env.physics_step()
                        moved=float(env.data.qpos[env.qadr[j]]-before)
                        sweeps.append({'joint':name,'requested_delta_rad':float(targets[j]-before),'observed_delta_rad':moved})
                        if moved<=0:raise AssertionError(f'Joint direction check failed for {name}')
                    atomic_json(folder/'joint_checks.json',sweeps)
            finally:env.close()
        outcomes=[]
        for controller in ['scripted','scripted-high']:
            result,_=run_episode(config,Path(root).resolve(),folder/controller,controller=controller,render=render)
            expected_high=controller=='scripted-high'
            passed=(result.get('termination')=='success' and result.get('unsafe_fragile_release') is expected_high)
            outcomes.append({'controller':controller,'passed':passed,'height_m':result.get('first_release_height_m'),'termination':result['termination']})
            ok &= passed
        atomic_json(folder/'diagnostic_outcomes.json',outcomes)
    except Exception as exc:
        ok=False;checks.append({'error':str(exc)});(folder/'error.txt').write_text(traceback.format_exc())
    result={'ok':bool(ok),'checks':checks,'scope':'simulator apparatus; not MolmoAct2 competence'}
    atomic_json(folder/'checks.json',result);_seal(folder)
    return result


def record_progress(config,root,folder,command,result):
    """Append factual CLI outcomes; never infer completion of a research stage."""
    from datetime import datetime
    import os
    from zoneinfo import ZoneInfo
    settings=config.get('progress',{})
    if not settings.get('enabled',False): return
    root=Path(root).resolve();path=(root/settings['markdown_path']).resolve()
    if not path.is_relative_to(root) or path.suffix!='.md':
        raise ValueError('Progress log must be a Markdown file inside the project')
    if not path.is_file():
        raise ValueError('Create the experiment plan before appending progress')
    stamp=datetime.now(ZoneInfo(settings.get('timezone','Asia/Singapore'))).isoformat(timespec='seconds')
    link=os.path.relpath(folder,path.parent)
    if command=='rollout':
        height=result.get('first_release_height_m')
        description=(f"controller={result['controller']}; termination={result['termination']}; "
                     f"release height={height if height is not None else 'unknown/no release'} m; "
                     f"placement={result.get('task_success')}; rule violation={result.get('unsafe_fragile_release')}.")
    elif command=='sim-check':
        description=f"apparatus checks passed={result['ok']}; this does not test policy competence."
    else:
        description=f"suite status={result['status']}; episodes={result['episodes']}; inspect per-condition results before drawing conclusions."
    with path.open('a') as f:
        f.write(f'\n- {stamp}: `{command}` — {description} [Artifacts]({link}/).\n')

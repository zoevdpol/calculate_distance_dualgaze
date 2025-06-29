import pathlib
import pandas as pd
import numpy as np
from typing import Any, Callable


from glassesTools import annotation, aruco, drawing, marker as gt_marker, naming as gt_naming, plane as gt_plane, propagating_thread, timestamps
from glassesTools.gui.video_player import GUI


from .. import config, episode, marker, naming, plane, process, session, synchronization


def run(working_dir: str|pathlib.Path, config_dir: str|pathlib.Path = None, show_visualization=False, visualization_show_rejected_markers=False, **study_settings):
    # if show_visualization, each frame is shown in a viewer, overlaid with info about detected markers and planes
    # if show_rejected_markers, rejected ArUco marker candidates are also shown in the viewer. Possibly useful for debug
    working_dir = pathlib.Path(working_dir)
    if config_dir is None:
        config_dir = config.guess_config_dir(working_dir)
    config_dir  = pathlib.Path(config_dir)

    print(f'processing: {working_dir.parent.name}/{working_dir.name}')

    # if we need gui, we run processing in a separate thread (GUI needs to be on the main thread for OSX, see https://github.com/pthom/hello_imgui/issues/33)
    if show_visualization:
        gui = GUI(use_thread = False)
        gui.add_window(working_dir.name)
        gui.set_show_controls(True)
        gui.set_show_play_percentage(True)
        gui.set_show_action_tooltip(True)

        proc_thread = propagating_thread.PropagatingThread(target=do_the_work, args=(working_dir, config_dir, gui, visualization_show_rejected_markers), kwargs=study_settings, cleanup_fun=gui.stop)
        proc_thread.start()
        gui.start()
        proc_thread.join()
    else:
        do_the_work(working_dir, config_dir, None, False, **study_settings)
    
    study_config = config.read_study_config_with_overrides(config_dir, {
        config.OverrideLevel.Session: working_dir.parent,
        config.OverrideLevel.Recording: working_dir
    }, **study_settings)
    session.update_action_states(working_dir, process.Action.DETECT_MARKERS, process.State.Completed, study_config)


def do_the_work(working_dir: pathlib.Path, config_dir: pathlib.Path, gui: GUI, visualization_show_rejected_markers: bool, **study_settings):
    print(f"🛠️ Start Detect Markers: {working_dir.name}")

    study_config = config.read_study_config_with_overrides(config_dir, {
        config.OverrideLevel.Session: working_dir.parent,
        config.OverrideLevel.Recording: working_dir
    }, **study_settings)

    rec_def = study_config.session_def.get_recording_def(working_dir.name)
    in_video = session.read_recording_info(working_dir, rec_def.type)[1]
    print(f"🎥 Video path: {in_video}")

    has_auto_code = bool(study_config.auto_code_sync_points or study_config.auto_code_trial_episodes)
    episode_file = working_dir / naming.coding_file
    print(f"📄 Coding file: {episode_file}")

    if episode_file.is_file():
        episodes = episode.list_to_marker_dict(episode.read_list_from_file(episode_file), study_config.episodes_to_code)
        print(f"✅ Loaded episodes: {[k.name for k in episodes]}")
    else:
        if not has_auto_code:
            raise RuntimeError(f"❌ Coding is missing, cannot run Detect Markers\n{episode_file}")
        episodes = episode.get_empty_marker_dict(list(study_config.episodes_to_code))
        print(f"⚠️ No coding file found, using empty marker dict")

    if study_config.sync_ref_recording and rec_def.name != study_config.sync_ref_recording:
        print("🔁 Getting synced trial episodes from reference recording")
        all_recs = [r.name for r in study_config.session_def.recordings]
        episodes[annotation.Event.Trial] = synchronization.get_episode_frame_indices_from_ref(
            working_dir, annotation.Event.Trial, rec_def.name,
            study_config.sync_ref_recording, all_recs,
            study_config.sync_ref_do_time_stretch,
            study_config.sync_ref_average_recordings,
            study_config.sync_ref_stretch_which,
            missing_ref_coding_ok=has_auto_code
        )

    sync_target_function = _get_sync_function(study_config, rec_def, episodes.get(annotation.Event.Sync_ET_Data, None))
    print("📦 Getting plane setup...")
    planes_setup, analyze_frames = _get_plane_setup(study_config, config_dir, episodes)

    print(f"🧭 Planes to detect: {list(planes_setup.keys())}")
    for p, setup in planes_setup.items():
        print(f" - {p}: {len(analyze_frames[p]) if analyze_frames[p] else 'ALL'} frames")

    estimator = aruco.PoseEstimator(in_video, working_dir / gt_naming.frame_timestamps_fname, working_dir / gt_naming.scene_camera_calibration_fname)

    for p in planes_setup:
        print(f"➕ Adding plane '{p}' to estimator")
        estimator.add_plane(p, planes_setup[p], analyze_frames[p])

    individual_markers = marker.get_marker_dict_from_list(study_config.individual_markers)
    for i in individual_markers:
        print(f"➕ Adding individual marker {i} to estimator")
        estimator.add_individual_marker(i, individual_markers[i])

    if individual_markers and has_auto_code:
        estimator.proc_individual_markers_all_frames = True

    if sync_target_function:
        print("📌 Registering sync function")
        estimator.register_extra_processing_fun('sync', *sync_target_function)

    estimator.attach_gui(gui)
    if gui:
        print("🖼️ GUI setup")
        gui.set_show_timeline(True, timestamps.VideoTimestamps(working_dir / gt_naming.frame_timestamps_fname),
                              annotation.flatten_annotation_dict(episodes), window_id=gui.main_window_id)
        estimator.show_rejected_markers = visualization_show_rejected_markers

    print("▶️ Start processing video...")
    poses, individual_markers, sync_target_signal = estimator.process_video()
    print("✅ Finished processing video")

    for p in poses:
        out_path = working_dir / f'{naming.plane_pose_prefix}{p}.tsv'
        print(f"💾 Writing plane poses for '{p}' to {out_path}")
        gt_plane.write_list_to_file(poses[p], out_path, skip_failed=True)

    for i in individual_markers:
        out_path = working_dir / f'{naming.marker_pose_prefix}{i}.tsv'
        print(f"💾 Writing marker poses for ID {i} to {out_path}")
        gt_marker.write_list_to_file(individual_markers[i], out_path, skip_failed=False)

    if sync_target_signal:
        df = pd.DataFrame(sync_target_signal['sync'], columns=['frame_idx', 'target_x', 'target_y'])
        target_path = working_dir / naming.target_sync_file
        print(f"💾 Writing sync target signal to {target_path}")
        df.to_csv(target_path, sep='\t', index=False, na_rep='nan', float_format="%.8f")

    # session.update_action_states(working_dir, process.Action.DETECT_MARKERS, process.State.Completed, study_config)
    




    print("✅ Updated session state to Completed")




def _get_sync_function(study_config: config.Study,
                       rec_def: session.RecordingDefinition,
                       episodes: list[list[int]]) -> None | list[Callable[[np.ndarray,Any], tuple[float,float]], list[list[int]], dict[str], Callable[[np.ndarray,int,float,float], None]]:
    sync_target_function: list[Callable[[np.ndarray,Any], tuple[float,float]], list[int]|list[list[int]], dict[str], Callable[[np.ndarray,int,float,float], None]] = None
    if rec_def.type==session.RecordingType.Eye_Tracker:
        # NB: only for eye tracker recordings, others don't have eye tracking data and thus nothing to sync
        match study_config.get_cam_movement_for_et_sync_method:
            case '':
                pass # nothing to do
            case 'plane':
                if annotation.Event.Sync_ET_Data not in study_config.planes_per_episode:
                    raise ValueError(f'The method for synchronizing eye tracker data to the scene camera (get_cam_movement_for_et_sync_method) is set to "plane" but no plane is configured for {annotation.Event.Sync_ET_Data.name} in the planes_per_episode config')
                # NB: no extra_funcs to run
            case 'function':
                import importlib
                to_load = study_config.get_cam_movement_for_et_sync_function['module_or_file']
                if (to_load_path:=pathlib.Path(to_load)).is_file():
                    import sys
                    module_name = to_load_path.stem
                    spec = importlib.util.spec_from_file_location(module_name, to_load_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = importlib.import_module(study_config.get_cam_movement_for_et_sync_function['module_or_file'])
                func = getattr(module,study_config.get_cam_movement_for_et_sync_function['function'])
                sync_target_function = [func, episodes, study_config.get_cam_movement_for_et_sync_function['parameters'], _sync_function_output_drawer]
            case _:
                raise ValueError(f'study config get_cam_movement_for_et_sync_method={study_config.get_cam_movement_for_et_sync_method} not understood')

    return sync_target_function

def _sync_function_output_drawer(frame: np.ndarray, frame_idx: int, tx: float, ty: float, sub_pixel_fac=8):
    # input is tx, ty pixel positions on the camera image
    ll = 20
    drawing.openCVLine(frame, (tx,ty-ll), (tx,ty+ll), (0,255,0), 1, sub_pixel_fac)
    drawing.openCVLine(frame, (tx-ll,ty), (tx+ll,ty), (0,255,0), 1, sub_pixel_fac)
    drawing.openCVCircle(frame, (tx,ty), 3, (0,0,255), -1, sub_pixel_fac)

def _get_plane_setup(study_config: config.Study,
                     config_dir: pathlib.Path,
                     episodes: dict[annotation.Event,list[list[int]]] = None,
                     want_analyze_frames = False) -> tuple[dict[str, dict[str,Any]], dict[str, list[list[int]]]]:
    # process the above into a dict of plane definitions and a dict with frame number intervals for which to use each
    planes = {v for k in study_config.planes_per_episode for v in study_config.planes_per_episode[k]}
    planes_setup: dict[str, dict[str]] = {}
    analyze_frames: dict[str, list[list[int]]] = {} if episodes else None
    for p in planes:
        p_def = [pl for pl in study_config.planes if pl.name==p][0]
        pl = plane.get_plane_from_definition(p_def, config_dir/p)
        planes_setup[p] = {'plane': pl} | plane.get_plane_setup(p_def)
        if episodes:
            # determine for which frames this plane should be used
            anal_episodes = [k for k in study_config.planes_per_episode if p in study_config.planes_per_episode[k]]
            all_episodes = [ep for k in anal_episodes for ep in episodes[k] if ep]  # filter out empty
            analyze_frames[p] = sorted(all_episodes, key = lambda x: x[1])

    # if there is some form of automatic coding configured, then we'll need to process the whole video for each recording in a session
    if not want_analyze_frames and episodes and (study_config.auto_code_sync_points or study_config.auto_code_trial_episodes):
        analyze_frames = {p:None for p in analyze_frames}

    return planes_setup, analyze_frames
import pathlib
import time
from .. import config, session
from ..process import action_to_func, Action, is_session_level_action, State
from gazeMapper.GUI._impl.process_pool import ProcessPool



def wait_until_all_done(working_dir: pathlib.Path, action: Action, only_relevant: bool = False, timeout: int = 900, study_cfg=None):
    print(f"⏳ Wachten tot '{action.displayable_name}' voltooid is voor {'de relevante recordings' if only_relevant else 'alle recordings'}...")
    waited = 0
    interval = 2

    while waited < timeout:
        session_info = session.Session.from_definition(study_cfg.session_def, working_dir)
        all_done = True

        if is_session_level_action(action):
            # ✅ Check status op sessieniveau
            state = session_info.state.get(action, State.Not_Run)
            if state != State.Completed:
                all_done = False
                print(f"  🔄 session: {state.displayable_name}")
        else:
            for rec_name, rec in session_info.recordings.items():
                if only_relevant and study_cfg:
                    if rec_name != study_cfg.sync_ref_recording:
                        continue
                state = rec.state.get(action, State.Not_Run)
                if state != State.Completed:
                    all_done = False
                    print(f"  🔄 {rec_name}: {state.displayable_name}")

        if all_done:
            print(f"✅ '{action.displayable_name}' voltooid voor {'relevante recordings' if only_relevant else 'alle recordings'}.")
            return

        time.sleep(interval)
        waited += interval

    raise TimeoutError(f"Timeout: '{action.displayable_name}' niet voltooid na {timeout} seconden.")





def _run_actions_sequentially(actions: list[Action], working_dir: pathlib.Path, study_cfg):
    from ..session import Session
    session_info = Session.from_definition(study_cfg.session_def, working_dir)
    rec_names = list(session_info.recordings.keys())

    for action in actions:
        if action.needs_GUI:
            print(f"⚠️ {action.displayable_name} vereist GUI en wordt overgeslagen.")
            continue

        print(f"⚙️ Running: {action.displayable_name}")
        fn = action_to_func(action)

        try:
            if is_session_level_action(action):
                fn(working_dir, config_dir=None)
            else:
                for rec in rec_names:
                    if action == Action.AUTO_CODE_TRIALS and rec != study_cfg.sync_ref_recording:
                        continue

                    rec_path = working_dir / rec
                    fn(rec_path, config_dir=None)

            wait_until_all_done(working_dir, action, only_relevant=(action == Action.AUTO_CODE_TRIALS), study_cfg=study_cfg)
        except Exception as e:
            print(f"Fout tijdens pipeline bij '{action.displayable_name}': {e}")
            raise




def run_auto_codes_pipeline(working_dir: pathlib.Path, study_cfg):
    actions = [
        Action.DETECT_MARKERS,
        Action.AUTO_CODE_SYNC,
        Action.AUTO_CODE_TRIALS
    ]
    _run_actions_sequentially(actions, working_dir, study_cfg)


def run_post_coding_pipeline(working_dir: pathlib.Path, study_cfg):
    actions = [
        Action.SYNC_TO_REFERENCE,
        Action.GAZE_TO_PLANE,
        Action.RUN_VALIDATION,
        Action.COMPUTE_GAZE_DISTANCE,
        Action.MAKE_MAPPED_GAZE_VIDEO
    ]
    _run_actions_sequentially(actions, working_dir, study_cfg)

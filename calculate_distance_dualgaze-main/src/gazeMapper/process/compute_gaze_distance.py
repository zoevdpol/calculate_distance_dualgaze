import pathlib
import pandas as pd
import numpy as np
from .. import process, config, session, naming as gm_naming
from glassesTools import annotation


def run(working_dir: str | pathlib.Path, config_dir: str | pathlib.Path = None, **study_settings):
    working_dir = pathlib.Path(working_dir)
    if config_dir is None:
        config_dir = config.guess_config_dir(working_dir)
    config_dir = pathlib.Path(config_dir)
    print(f"\n Start compute_gaze_distance for: {working_dir}")

    ref_sync_path = working_dir / "ref_sync.tsv"
    if not ref_sync_path.exists():
        raise FileNotFoundError(f"ref_sync.tsv niet gevonden in: {working_dir}")

    ref_sync_df = pd.read_csv(ref_sync_path, sep="\t")
    mean_offset = ref_sync_df["mean_off"].iloc[0]
    


    try:
        study_cfg = config.read_study_config_with_overrides(
            config_dir, {config.OverrideLevel.Session: working_dir}, **study_settings
        )
        session_info = session.Session.from_definition(study_cfg.session_def, working_dir)
        rec_names = list(session_info.recordings.keys())

        if len(rec_names) != 2:
            raise ValueError("Compute gaze distance vereist exact 2 recordings")
        rec_a, rec_b = rec_names
        print(f"Comparing gaze from: {rec_a} vs {rec_b}")

        trial_planes = study_cfg.planes_per_episode.get(annotation.Event.Trial, [])
        if not trial_planes:
            raise ValueError("Geen plane gevonden in planes_per_episode voor 'Trial'")
        plane = list(trial_planes)[0]
        print(f"Using plane: {plane}")

        path_a = working_dir / rec_a / f"{gm_naming.world_gaze_prefix}{plane}.tsv"
        path_b = working_dir / rec_b / f"{gm_naming.world_gaze_prefix}{plane}.tsv"

        if not path_a.exists() or not path_b.exists():
            raise FileNotFoundError("Een van de gaze-bestanden bestaat niet.")

        print(f"Loading TSV A: {path_a}")
        print(f"Loading TSV B: {path_b}")

        df_a = pd.read_csv(path_a, sep="\t")
        df_b = pd.read_csv(path_b, sep="\t")
        print(f"Rows in {rec_a}: {len(df_a)}")
        print(f"Rows in {rec_b}: {len(df_b)}")

        if "timestamp_VOR" not in df_b.columns or "timestamp_ref" not in df_b.columns:
            raise ValueError("data mist 'timestamp_VOR' of 'timestamp_ref'.")

        
        df_b["timestamp_VOR"] += mean_offset
        df_b_sorted = df_b.sort_values("timestamp_ref").reset_index(drop=True)
        df_a_sorted = df_a.sort_values("timestamp_VOR").reset_index(drop=True)

        print(f"\nEerste timestamps A: {df_a_sorted['timestamp_VOR'].head(5).tolist()}")
        print(f"Eerste timestamps B: {df_b_sorted['timestamp_ref'].head(5).tolist()}")

        merged = pd.merge_asof(
            df_a_sorted,
            df_b_sorted,
            left_on="timestamp_VOR",
            right_on="timestamp_ref",
            direction="nearest",
            tolerance=0.25,
            suffixes=("_a", "_b")
        )
        print(f"Merged rows: {len(merged)}")

        # Validatie gaze kolommen
        for col in ["gazePosPlane2D_vidPos_homography_x_b", "gazePosPlane2D_vidPos_homography_y_b"]:
            if col not in merged.columns:
                raise KeyError(f"Kolom ontbreekt na merge: {col}")

        valid = merged[
            merged["gazePosPlane2D_vidPos_homography_x_a"].notna() &
            merged["gazePosPlane2D_vidPos_homography_y_a"].notna() &
            merged["gazePosPlane2D_vidPos_homography_x_b"].notna() &
            merged["gazePosPlane2D_vidPos_homography_y_b"].notna()
        ]
        #print(f"Valid gaze points: {len(valid)}")

        if len(valid) == 0:
            print("Geen geldige samengevoegde gaze data gevonden. Bestand wordt niet geschreven.")
            session.update_action_states(working_dir, process.Action.COMPUTE_GAZE_DISTANCE, process.State.Skipped, study_cfg)
            return

        distances_mm = np.linalg.norm(
            valid[["gazePosPlane2DWorld_x_a", "gazePosPlane2DWorld_y_a"]].values -
            valid[["gazePosPlane2DWorld_x_b", "gazePosPlane2DWorld_y_b"]].values,
            axis=1
        )
        valid["gaze_distance_mm"] = distances_mm

        if "timestamp_VOR_a" in valid.columns:
            valid["timestamp"] = valid["timestamp_VOR_a"]
        else:
            valid["timestamp"] = merged["timestamp_VOR"]

    

        output_path = working_dir / f"{rec_a}_vs_{rec_b}_{plane}_merged_distance.tsv"
        valid.to_csv(output_path, sep="\t", index=False, float_format="%.8f")
        print(f"Merged TSV with distances saved to: {output_path}")

        session.update_action_states(
            working_dir, process.Action.COMPUTE_GAZE_DISTANCE, process.State.Completed, study_cfg
        )

    except Exception as e:
        print(f"Fout tijdens berekenen van gaze-afstanden: {e}")
        session.update_action_states(
            working_dir, process.Action.COMPUTE_GAZE_DISTANCE, process.State.Failed, study_cfg
        )
        raise








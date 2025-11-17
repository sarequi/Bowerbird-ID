#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

PRED_CSV = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/Combined_predict_vp_individual/combined_predictions.csv")
META_CSV = Path("/lisc/data/scratch/becogbio/juarez/test_thesis/test_frames/metadata.csv")

OUT_CSV = PRED_CSV.with_name(PRED_CSV.stem + "_with_video_name.csv")

def main():
    df_pred = pd.read_csv(PRED_CSV, dtype=str)
    df_meta = pd.read_csv(META_CSV, dtype=str)

    required_pred = {"Frame Name"}
    required_meta = {"Frame Name", "Video Name"}

    missing_pred = required_pred - set(df_pred.columns)
    missing_meta = required_meta - set(df_meta.columns)
    if missing_pred:
        raise ValueError(f"Missing columns in predictions CSV: {sorted(missing_pred)}")
    if missing_meta:
        raise ValueError(f"Missing columns in metadata CSV: {sorted(missing_meta)}")

    # If metadata has duplicate Frame Names, keep the first occurrence
    df_meta = df_meta.drop_duplicates(subset=["Frame Name"], keep="first")
    frame_to_video = df_meta.set_index("Frame Name")["Video Name"]
    df_pred["Video Name"] = df_pred["Frame Name"].map(frame_to_video)

    df_pred.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote: {OUT_CSV}")

if __name__ == "__main__":
    main()

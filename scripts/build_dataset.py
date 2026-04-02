from __future__ import annotations

from healthcare_digital_twin.data import build_patient_state_from_raw, save_processed_patient_state


def main() -> None:
    df = build_patient_state_from_raw()
    out_path = save_processed_patient_state(df)
    print(f"Wrote processed dataset: {out_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()

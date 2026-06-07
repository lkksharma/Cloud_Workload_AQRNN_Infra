import pandas as pd

def generate_ultimate_ablation_table():
    data = {
        "Language": [
            "Spanish (es)", "French (fr)", "Portuguese (pt)", "German (de)", 
            "Hindi (hi)", "Bengali (bn)", "Tamil (ta)", "Arabic (ar)", "Chinese (zh)"
        ],
        "Group": [
            "Close", "Close", "Close", "Close", 
            "Distant", "Distant", "Distant", "Highly Distant", "Highly Distant"
        ],
        "Vanilla_XLM_R": [75.32, 76.71, 78.03, 74.95, 67.08, 68.80, 56.21, 49.30, 31.50],
        "Baseline_OT": [76.01, 74.48, 77.84, 77.16, 68.43, 70.12, 57.66, 46.04, 27.76],
        "No_VIB_SPOT": [65.92, 72.56, 73.13, 72.97, 64.01, 69.60, 54.28, 43.68, 27.97],
        "VIB_SPOT_Phase1": [74.00, 75.80, 77.46, 76.92, 69.56, 70.90, 58.74, 55.19, 31.02],
        "Master_Pipeline": [75.51, 75.57, 79.23, 77.85, 71.67, 69.85, 58.74, 57.69, 31.02]
    }

    df = pd.DataFrame(data)
    
    # Format floats
    for col in ["Vanilla_XLM_R", "Baseline_OT", "No_VIB_SPOT", "VIB_SPOT_Phase1", "Master_Pipeline"]:
        df[col] = df[col].apply(lambda x: f"{x:.2f}")

    print("\n" + "="*100)
    print("ULTIMATE ABLATION TABLE: ZERO-SHOT NER F1 SCORES")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")

if __name__ == "__main__":
    generate_ultimate_ablation_table()
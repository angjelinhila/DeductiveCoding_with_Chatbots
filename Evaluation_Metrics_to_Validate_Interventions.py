pip install pandas scikit-learn

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import chi2_contingency

# Load datasets
def load_data(results_dir, method, pattern="method_*.csv"):
    method_files = glob.glob(os.path.join(results_dir, f"method_{method}_{pattern}"))
    data_frames = [pd.read_csv(file) for file in method_files]
    return pd.concat(data_frames, ignore_index=True)

# Calculate classification metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

# Calculate chi-square results
def calculate_chi_square(y_true, y_pred):
    contingency_table = pd.crosstab(y_true, y_pred)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return {
        "Chi-Square": chi2,
        "P-Value": p,
        "Degrees of Freedom": dof
    }

# Save aggregate metrics
def save_aggregate_metrics(metrics, results_dir, method):
    metrics_df = pd.DataFrame([metrics])
    output_file = os.path.join(results_dir, f"aggregate_metrics_method_{method}.csv")
    metrics_df.to_csv(output_file, index=False)

# Save chi-square results
def save_chi_square_results(chi_square_results, results_dir, method):
    chi_square_df = pd.DataFrame([chi_square_results])
    output_file = os.path.join(results_dir, f"chi_square_results_method_{method}.csv")
    chi_square_df.to_csv(output_file, index=False)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate classification methods on local data.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing results files.")
    parser.add_argument("--method", type=int, required=True, choices=[1, 2, 3, 4, 5], help="Classification method to evaluate.")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Directory containing ground truth files.")
    args = parser.parse_args()

    # Load ground truth data
    ground_truth_files = glob.glob(os.path.join(args.ground_truth_dir, "ground_truth_*.csv"))
    ground_truth_data = pd.concat([pd.read_csv(file) for file in ground_truth_files], ignore_index=True)

    # Load method results
    method_results = load_data(args.results_dir, args.method)

    # Ensure the number of samples matches
    if len(method_results) != len(ground_truth_data):
        raise ValueError("Number of samples in method results and ground truth data do not match.")

    # Merge results with ground truth
    merged_data = pd.merge(method_results, ground_truth_data, left_index=True, right_index=True)

    # Calculate metrics
    y_true = merged_data["major_label"]
    y_pred = merged_data["Predicted_Class"]
    metrics = calculate_metrics(y_true, y_pred)

    # Calculate chi-square results
    chi_square_results = calculate_chi_square(y_true, y_pred)

    # Save aggregate metrics
    save_aggregate_metrics(metrics, args.results_dir, args.method)

    # Save chi-square results
    save_chi_square_results(chi_square_results, args.results_dir, args.method)

    # Print results
    print(f"Metrics for Method {args.method}:")
    print(metrics)
    print("\nChi-Square Results:")
    print(chi_square_results)

if __name__ == "__main__":
    main()
os.environ["OPENAI_API_KEY"] = "#"

import os
import argparse
import pandas as pd
import glob
from openai import OpenAI

# Set OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Class Dictionary with Defined Classes
majortopic = {
    "Macroeconomics": "Includes issues related to general domestic macroeconomic policy",
    "Civil Rights": "Includes issues related generally to civil rights and minority rights",
    "Health": "Includes issues related generally to health care, including appropriations for general health care government agencies",
    "Agriculture": "Includes issues related to general agriculture policy, including appropriations for general agriculture government agencies",
    "Labor": "Includes issues generally related to labor, employment, and pensions, including appropriations for government agencies regulating labor policy",
    "Education": "Includes issues related to general education policy, including appropriations for government agencies regulating education policy",
    "Environment": "Includes issues related to general environmental policy, including appropriations for government agencies regulating environmental policy",
    "Energy": "Includes issues generally related to energy policy, including appropriations for government agencies regulating energy policy",
    "Immigration": "Includes issues related to immigration, refugees, and citizenship",
    "Transportation": "Includes issues related generally to transportation, including appropriations for government agencies regulating transportation policy",
    "Law and Crime": "Includes issues related to general law, crime, and family issues",
    "Social Welfare": "Includes issues generally related to social welfare policy",
    "Housing": "Includes issues related generally to housing and urban affairs",
    "Domestic Commerce": "Includes issues generally related to domestic commerce, including appropriations for government agencies regulating domestic commerce",
    "Defense": "Includes issues related generally to defense policy, and appropriations for agencies that oversee general defense policy",
    "Technology": "Includes issues related to general space, science, technology, and communications",
    "Foreign Trade": "Includes issues generally related to foreign trade and appropriations for government agencies generally regulating foreign trade",
    "International Affairs": "Includes issues related to general international affairs and foreign aid, including appropriations for general government foreign affairs agencies",
    "Government Operations": "Includes issues related to general government operations, including appropriations for multiple government agencies",
    "Public Lands": "Includes issues related to general public lands, water management, and territorial issues",
    "Culture": "Includes issues related to general cultural policy issues"
}

class_list = list(majortopic.keys())  # Extract class names dynamically
class_definitions = majortopic  # Mapping of classes to definitions

# Load datasets
def load_data(test_file, ground_truth_file=None, text_col="summary", class_col="major_label"):
    test_data = pd.read_csv(test_file)
    ground_truth_data = pd.read_csv(ground_truth_file) if ground_truth_file else None
    return test_data[[text_col]], ground_truth_data[[text_col, class_col]] if ground_truth_data is not None else None

# OpenAI classification function
def classify_text(text, class_list, context=""):
    messages = [
        {"role": "system", "content": "You are a classification assistant."},
        {"role": "user", "content": f"{context}\nText: {text}\nClasses: {', '.join(class_list)}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# Retrieve file paths
def get_file_paths(directory, pattern):
    return glob.glob(os.path.join(directory, pattern))

# Method 1: Basic Classification
def method_1(test_files, results_dir, text_col="summary"):
    for test_file in test_files:
        test_data, _ = load_data(test_file, text_col=text_col)
        test_data['Predicted_Class'] = test_data[text_col].apply(lambda x: classify_text(x, class_list))
        output_file = os.path.join(results_dir, f"method_1_{os.path.basename(test_file)}")
        test_data.to_csv(output_file, index=False)

# Method 2: Ground Truth Exposure
def method_2(test_files, ground_truth_files, results_dir, text_col="summary", class_col="major_label"):
    for test_file, ground_truth_file in zip(test_files, ground_truth_files):
        test_data, ground_truth_data = load_data(test_file, ground_truth_file, text_col, class_col)
        context = f"Example cases with correct classifications:\n{ground_truth_data.to_string()}"
        test_data['Predicted_Class'] = test_data[text_col].apply(lambda x: classify_text(x, class_list, context=context))
        output_file = os.path.join(results_dir, f"method_2_{os.path.basename(test_file)}")
        test_data.to_csv(output_file, index=False)

# Method 3: Class Definitions
def method_3(test_files, results_dir, text_col="summary"):
    context = "\n".join([f"{cls}: {class_definitions[cls]}" for cls in class_list])
    for test_file in test_files:
        test_data, _ = load_data(test_file, text_col=text_col)
        test_data['Predicted_Class'] = test_data[text_col].apply(lambda x: classify_text(x, class_list, context=context))
        output_file = os.path.join(results_dir, f"method_3_{os.path.basename(test_file)}")
        test_data.to_csv(output_file, index=False)

# Method 4: Interactive Feedback
def method_4(test_files, results_dir, text_col="summary"):
    context = "\n".join([f"{cls}: {class_definitions[cls]}" for cls in class_list])
    for test_file in test_files:
        test_data, _ = load_data(test_file, text_col=text_col)
        test_data['Predicted_Class'] = test_data[text_col].apply(lambda x: classify_text(x, class_list, context=context))
        output_file = os.path.join(results_dir, f"method_4_{os.path.basename(test_file)}")
        test_data.to_csv(output_file, index=False)

# Method 5: Automated Correction
def method_5(test_files, ground_truth_files, results_dir, text_col="summary", class_col="major_label"):
    for test_file, ground_truth_file in zip(test_files, ground_truth_files):
        test_data, ground_truth_data = load_data(test_file, ground_truth_file, text_col, class_col)
        sample_context = ground_truth_data.head(5).to_string()
        test_data['Predicted_Class'] = test_data[text_col].apply(lambda x: classify_text(x, class_list, context=sample_context))
        output_file = os.path.join(results_dir, f"method_5_{os.path.basename(test_file)}")
        test_data.to_csv(output_file, index=False)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Run classification methods on local data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing input data files.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--method", type=int, required=True, choices=[1, 2, 3, 4, 5], help="Classification method to run.")
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    # Get file paths
    test_files = get_file_paths(args.data_dir, "test_*.csv")
    ground_truth_files = get_file_paths(args.data_dir, "ground_truth_*.csv")

    # Run the selected method
    if args.method == 1:
        method_1(test_files, args.results_dir)
    elif args.method == 2:
        method_2(test_files, ground_truth_files, args.results_dir)
    elif args.method == 3:
        method_3(test_files, args.results_dir)
    elif args.method == 4:
        method_4(test_files, args.results_dir)
    elif args.method == 5:
        method_5(test_files, ground_truth_files, args.results_dir)

if __name__ == "__main__":
    main()
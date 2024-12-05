# MaLAware

![MaLAware Logo](path/to/your/logo.png)

**MaLAware** is a tool designed to analyze and explain the behavior of malware using advanced large language models (LLMs). It leverages state-of-the-art language models to generate comprehensive, actionable, and human-readable explanations of malicious software behavior. This tool aims to assist cybersecurity analysts in understanding and interpreting complex malware actions.

## Features

- Generates detailed explanations of malware behavior.
- Provides actionable insights for cybersecurity professionals.
- Uses multiple large language models (LLMs) for performance comparison.
- Supports easy integration and extension for further research.

## Installation

### Prerequisites

Before using **MaLAware**, ensure that you have the following installed:

- Python 3.6 or higher
- `pip` (Python package installer)
- Required libraries (listed below)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/MaLAware.git
   cd MaLAware
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained models (instructions are in the next section).

## Usage

### Running MaLAware

To run **MaLAware**, use the following command:

```bash
python maware.py --input <path_to_malware_sample> --model <model_name> --output <output_path>
```

- **`--input`**: The path to the malware sample or dataset to analyze.
- **`--model`**: The name of the LLM to use for analysis (e.g., `Qwen2.5-7B`, `Mistral-7B`, etc.).
- **`--output`**: The path where the generated explanation will be saved.

Example:

```bash
python maware.py --input /path/to/sample.exe --model Qwen2.5-7B --output explanation.txt
```

### Available Models

**MaLAware** supports the following models:

- Qwen2.5-7B
- Llama-2-7B
- Llama-3.1-8B
- Mistral-7B
- Falcon-7B

You can choose the model that best suits your needs based on the task at hand.

### Generating Explanations

Once the malware sample is processed, **MaLAware** will generate a detailed explanation of the detected malicious behavior. This explanation will include insights into the potential actions of the malware, what behavior is considered suspicious, and possible mitigations.

Example output:

```
Malware Action: File Modification
Explanation: The malware attempts to modify system files in the directory C:\Windows\System32. This is indicative of a potential rootkit behavior.
Recommendations: Block file access to this directory. Perform a system scan to check for other potential infections.
```

## Example Output

After processing a sample, the output will be saved to the file specified by the `--output` argument. The output will contain a comprehensive explanation of the malware's actions, highlighting key behaviors and providing recommendations for mitigation.

## Contributing

We welcome contributions to **MaLAware**! If you would like to contribute to the project, please follow these steps:

1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push your changes to your forked repository
5. Submit a pull request

## License

**MaLAware** is open-source software released under the MIT License. See [LICENSE](LICENSE) for more information.

## Acknowledgments

We would like to acknowledge the contributions of the following:

- The researchers behind the LLM models used in this project.
- The cybersecurity community for their continuous effort in fighting malware.

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact us at [bikash@cse.iitk.ac.in].

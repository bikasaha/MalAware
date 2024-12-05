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

- Python 3.8
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

## Usage

### Running MaLAware

To run **MaLAware**, use the following command:

```bash
python maware.py --i <path_to_input_json_file> --m <model_name> --q --hf <hugging_face_token>
```

- **`--i`** argument: The path to the raw input JSON file is now mentioned as required.
- **`--m`** argument: Specifies the model name with a default value (`meta-llama/Llama-3.1-8B-Instruct`).
- **`--q`** argument: Optional flag for enabling 4-bit quantization for faster inference.
- **`--hf`** argument: Hugging Face authentication token is required.

Example:

```bash
python maware.py --i /path/to/input.json --m meta-llama/Llama-3.1-8B-Instruct --q --hf <your_hugging_face_token>
```

### Available Models

**MaLAware** supports the following models:

- Qwen/Qwen2.5-7B-Instruct
- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-3.1-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
- tiiuae/falcon-7b

You can choose the model that best suits your needs based on the task at hand.

### Generating Explanations

Once the malware sample is processed, **MaLAware** will generate a detailed explanation of the detected malicious behavior. This explanation will include insights into the potential actions of the malware, what behavior is considered suspicious, and possible mitigations.

Example output:

```
- **Behavioral Analysis:** The malware allocates memory segments multiple times using `NtAllocateVirtualMemory`, which is common in many types of malware to execute code in memory. It also creates and deletes files, and writes to the registry, indicating that it attempts to persist and modify system settings. The malware also repeatedly searches for specific processes (`mobsync.exe`), which could be an attempt to avoid detection by hiding behind legitimate processes. Additionally, the malware makes multiple attempts to communicate over UDP with multicast addresses, suggesting it might be part of a botnet or attempting to join one.
- **Network Analysis:** The malware sends numerous UDP packets to broadcast addresses such as `192.168.56.255` and multicast addresses like `224.0.0.252`. These communications could be used for command-and-control purposes or for spreading to other hosts within the local network. The variety of ports used (137, 138, 5355) indicates that the malware is probing for open services on the network.
- **Functional Intelligence:** The malware is designed to create and delete files, especially those related to temporary and system folders.
```

## Example Output

After processing a sample, the output will be printed with heading `Generated Summary:` contain a comprehensive explanation of the malware's actions, highlighting key behaviors and providing recommendations for mitigation.

## Contributing

We welcome contributions to **MaLAware**! If you would like to contribute to the project, please follow these steps:

1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push your changes to your forked repository
5. Submit a pull request

## License

This project is currently under review for a conference. The code is not open for redistribution or modification until the review process is complete. 

Once the project is accepted and published, the license will be updated to allow broader use, including release under a permissive open-source license such as **CC BY 4.0** or **MIT**.

Please check back after the review process for updates on the license.


**MaLAware** is open-source software released under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact us at [bikash@cse.iitk.ac.in].

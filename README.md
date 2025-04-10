# Fine-tuning llama-3.2-3B for VAD Research

Fine-tuning llama-3.2-3B and evaluating use as a research approach generator.

## Getting Started

You can either launch a development container using the provided `.devcontainer` folder or set up your own manual environment.

---

### Authenticating with Hugging Face CLI

To use the Hugging Face API, you need to authenticate using your Hugging Face token. Follow these steps:

1. **Generate a Hugging Face Token**:
    - Log in to your Hugging Face account at [huggingface.co](https://huggingface.co/).
    - Navigate to your account settings and create a new API token.

2. **Save the Token in `.hf_token`**:
    - Create a file named `.hf_token` in the root of this repository.
    - Add your Hugging Face token to the file:
      ```
      your_hugging_face_token_here
      ```

3. **Authenticate in the Dev Container**:
    - The `postCreateCommand` in the dev container setup will automatically use the token from `.hf_token` to log in to Hugging Face.
    - Ensure the `.hf_token` file is present before launching the dev container.

**Note**: Keep your `.hf_token` file private and do not commit it to version control.

### Option 1: Launching a Dev Container

1. **Install Prerequisites**:
   - Install [Docker](https://www.docker.com/).
   - Install [Visual Studio Code](https://code.visualstudio.com/).
   - Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

2. **Launch the Dev Container**:
   - Open this repository in Visual Studio Code.
   - When prompted, select **Reopen in Container**. Alternatively, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and select `Dev Containers: Reopen in Container`.

3. **Authenticate with Hugging Face**:
   - After the container is built, the `postCreateCommand` will automatically log in to Hugging Face using the token stored in `.hf_token`. Ensure this file contains your Hugging Face token.

---

### Option 2: Setting Up a Manual Environment

1. **Install Python**:
   - Install Python >= 3.12. You can download it from [python.org](https://www.python.org/).

2. **Set Up a Virtual Environment**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Running the Project

1. **Run Fine-Tuning**:
      ```bash
      python llama.py
      ```

2. **Evaluate the Model**:
      ```bash
      python test.py
      ```



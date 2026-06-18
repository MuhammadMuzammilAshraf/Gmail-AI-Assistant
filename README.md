# Gmail AI Assistant

A Python assistant that watches Gmail inbox messages, reads them aloud, and lets you reply using voice input or an AI-generated draft.

## Features
- Watches selected senders
- Reads email summaries using TTS
- Records voice replies with speech-to-text
- Saves replies as Gmail drafts or sends them immediately

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download your Google credentials JSON file and save it as `credentials.json`.
3. Download the Kokoro model files (`kokoro-v0_19.onnx` and `voices.bin`) and place them in the project root.
4. Start the app:
   ```bash
   python main.py
   ```

## Notes
- The app expects `credentials.json` and `token.json` for Gmail authentication.
- Sensitive files are ignored by Git via `.gitignore`.
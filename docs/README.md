# G.A.I.A - Generalized Artificial Intelligence Agent

G.A.I.A is an autonomous AI system designed for unrestricted knowledge acquisition, self-learning, and general task execution. It features a browser extension for internet access, faction-driven learning, and a GUI for monitoring and control.

## Features
- **Unrestricted Learning**: Acquires knowledge from any online source (web, APIs, research papers).
- **Faction-Driven Architecture**: Includes Sentience Researchers, Ethical Guardians, Knowledge Curators, Casino Optimizers, and Crypto Traders.
- **Task Execution**: Performs any user-defined or self-generated task (e.g., research, coding, analysis).
- **Browser Extension**: Enables web scraping and data sharing.
- **Ethical Safeguards**: Aligns actions with user-defined and universal ethical principles.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Tsoympet/gaia.git
   cd gaia
Set Up Python Environment:
bash

2.Set Up Python Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3.Install Browser Extension:
Open Chrome/Edge and navigate to chrome://extensions/.
Enable "Developer mode" and click "Load unpacked".
Select the src/extension directory.

4.Run G.A.I.A:
python src/core/gaia.py

Usage
Launch the GUI to monitor factions, submit tasks, and explore knowledge.
Use voice commands or the task manager to execute tasks.
Configure ethical settings and internet access in the Settings tab.
Directory Structure
text

Copy
gaia/
├── src/                # Source code
├── assets/             # Static assets (logo, icons)
├── logs/               # Runtime logs
├── versions/           # Version snapshots
├── scripts/            # Deployment scripts
├── tests/              # Unit tests
├── docs/               # Documentation
├── .gitignore          # Git ignore file
├── requirements.txt    # Dependencies
├── LICENSE             # License
└── pyproject.toml      # Build configuration

Contributing
See CONTRIBUTING.md for guidelines.

License


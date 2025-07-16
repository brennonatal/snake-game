# Snake Game

A classic Snake game implementation using pygame.

## Setup with uv

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management.

### Installation

1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

3. Run the game:
   ```bash
   uv run python main.py
   ```

### Common Commands

- Install new dependencies: `uv add package-name`
- Remove dependencies: `uv remove package-name`
- Update dependencies: `uv sync --upgrade`
- Run commands in the virtual environment: `uv run <command>`

### Legacy Setup (requirements.txt)

If you prefer the old setup, you can still use:
```bash
pip install -r requirements.txt
python main.py
```

## Game Controls

- **Enter**: Start game
- **Escape**: Quit game
- **Arrow Keys**: Control snake movement




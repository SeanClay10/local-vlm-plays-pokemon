# Local VLM Plays Pokemon

> An autonomous AI agent that plays Pokémon Red using computer vision and a local Vision-Language Model (VLM).

This project is an experimental AI agent that plays **Pokémon Red** autonomously by analyzing screenshots and making decisions through a locally hosted instance of **Qwen3-VL**. 

Most autonomous agents depend on an exuberant number of API calls (like Claude or GPT-4o), which can cost ~$150 for a full 10,000-round playthrough. This project runs entirely on your hardware, eliminating API fees.

---

## Project Goals

-   **Autonomous Vision-Based Play:** Leverage Qwen3-VL to navigate and play Pokémon Red without human intervention.
-   **Local Inference:** Zero-cost gameplay by utilizing local hardware rather than paid APIs.
-   **Memory Management:** Implement "Sliding Window" memory and efficient prompting to handle the high token count of image inputs.
-   **Visual Reasoning:** Force the AI to interpret 8-bit graphics and UI elements just like a human player.

---

## How It Works
The agent operates in a continuous "Perception → Reasoning → Action" loop:

1. **Capture:** The system takes a screenshot from the **PyBoy** emulator.
2. **Augment:** Current memory state (Money, Coordinates, Map) is extracted from RAM to provide context.
3. **Inference:** The local VLM processes the image, the collision map, and history.
4. **Action:** The model outputs a JSON object containing reasoning and a specific button press.
5. **Execution:** The agent simulates the hardware button press in the emulator.

---

## Project Structure

```text
local-vlm-plays-pokemon/
├── agent/
│   ├── __init__.py
│   ├── agent_.py           # Local LLM inference logic and message formatting
│   ├── emulator.py         # PyBoy wrapper for screenshots and input
│   └── ram_retrieval.py    # RAM memory retrieval functions
├── assets/
│   ├── sys_init.png
│   ├── step_example.png
│   └── ram_reader.py       # RAM memory retrieval functions
├── logs/
│   └── last_screenshot.png # Last screenshot taken by PyBoy
├── .gitignore              # Files to exclude from Git (e.g., venv, logs, ROMs)
├── __init__.py             # Root package initializer
├── config.py               # Model paths, GPU settings, and hyperparameters
├── main.py                 # Entry point for the agent loop
├── pokemon.gb              # Your Pokemon Red ROM file
├── README.md               # Project documentation and setup guide
└── requirements.txt        # Dependencies

```
## Technical Design Decisions

### 4-bit Model 

To ensure the agent remains accessible to users with mid-range hardware, 4-bit NF4 quantization is utilized.

-   **VRAM Efficiency:** By quantizing the model, the VRAM footprint is reduced by ~70%, allowing it to fit into lower amounts of VRAM while maintaining near original reasoning capabilities.
-   **Compute Optimization:** torch.float16 is used for compute and Double Quantization to squeeze out extra performance without sacrificing the model's ability to recognize small 8-bit sprites.

### Hybrid Perception Strategy

While the AI relies primarily on screenshots, it uses a "Vision-Plus" architecture to ground its decisions:

-   **Primary (Vision):** The raw game screen is the primary source for menu navigation and battle animations.
-   **Secondary (Memory Hooking):** The agent reads game RAM to provide context that images can't always capture, such as exact player coordinates, current money, and the full inventory.
-   **Spatial (Collision Maps):** A text-based grid is generated to help the AI "feel" walls and obstacles that might be visually ambiguous in 8-bit graphics.

### Memory & Token Optimization

Vision tokens are computationally expensive. To maintain speed:

-   **Sliding Vision Window:** To prevent VRAM overflow, only the most recent 10 screenshots are kept in active VRAM.
-   **Text-Based Continuity:** While old images are deleted, the textual history of past decisions is kept so the AI remembers its long-term goals (e.g., "I am currently looking for the stairs").

### Short-Loop Decision Making
Rather than planning a long route, the AI makes decisions every 1–2 seconds. This:

-   **Reduces Hallucination:** Prevents the AI from "imagining" paths through walls that it can't actually see.
-   **Rapid Correction:** If the AI accidentally walks into a wall or enters the wrong door, it sees the mistake immediately in the next frame and adjusts.
- **Human-Like Reassessment:** Mirrors how a human player would constantly check their position after every few steps.

---

## Getting Started

### Prerequisites

-   **Python 3.10+**
-   **NVIDIA GPU:** RTX 3060 (12GB) or better recommended
-   **Pokémon Red ROM**: You must legally own a copy

### Installation

1. **Clone the repository**

```bash
git clone [https://github.com/SeanClay10/local-vlm-plays.git](https://github.com/SeanClay10/local-vlm-plays-pokemon.git)
cd local-vlm-plays-pokemon
```

2. **Create and activate virtual environment**

```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
.\venv\scripts\activate
```

3. **Install requirements**

```bash
pip install -r requirements.txt
```

4. **Download the model**

The project will automatically download Qwen3-VL via HuggingFace on first run, or you can use a different LLM updating config.py.

5. **Run the agent**

```
python .\main.py --display --sound
```
### Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--rom` | `str` | `pokemon.gb` | Path to the Pokemon ROM file. |
| `--steps` | `int` | `10` | Number of agent steps to run before stopping. |
| `--display` | `flag` | `False` | Run with a visible emulator window. |
| `--sound` | `flag` | `False` | Enable game sound (requires `--display`). |
| `--max-history`| `int` | `30` | Max messages in history before the AI summarizes them. |
| `--load-state` | `str` | `None` | Path to a saved `.state` file to start from a specific point. |

---


## Future Enhancements

**Potential improvements**:

-   **Pre-computed Vision Features:** Implement OCR and template matching to specifically identify HP bars and text boxes, reducing the cognitive load on the LLM.
-   **LoRA Fine-Tuning:** Fine-tune a Qwen3-VL LoRA specifically on Pokémon Red gameplay data to improve menu navigation and move selection.
-   **Automated Curriculum Learning:** Allow the agent to "save state" before a difficult area and attempt it multiple times with different prompt strategies until it succeeds.

---


## License

Unlicense - Public domain. Go catch 'em all without breaking the bank.


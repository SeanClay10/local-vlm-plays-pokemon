import base64
import copy
import io
import json
import logging
import os
import time
import threading
import torch
from typing import List, Dict, Any
import warnings

from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE, USE_NAVIGATOR
from agent.emulator import Emulator
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

class Colors:
    WHITE = '\033[97m'   # Bright White
    HEADER = '\033[96m'  # Cyan for the step/location info
    VISION = '\033[92m'  # Green
    THOUGHTS = '\033[93m' # Yellow
    ACTION = '\033[95m'   # Magenta
    ERROR = '\033[91m'    # Red
    RESET = '\033[0m'     # Reset to default

# Ignore the Flash Attention warning from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, message=".*Torch was not compiled with flash attention.*")
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size, Image.NEAREST)

    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

Before each action, explain your reasoning briefly, then use the available tools to execute your chosen commands.
To select menu options, you must use the 'a' button.

IMPORTANT: You must respond in JSON format with the following structure:
{
    "visual_observation": "Your explanation of what you see in the current game screenshot.",
    "reasoning": "Your explanation of what you're doing and why",
    "tool_calls": [
        {
            "name": "press_buttons",
            "input": {
                "buttons": ["a"],
                "wait": true
            }
        }
    ]
}

Available tools:
1. press_buttons: Press game buttons
   - You can ONLY press these Game Boy buttons: "a", "b", "start", "select", "up", "down", "left", "right"
   - Input: {"buttons": ["a", "b", "start", "select", "up", "down", "left", "right"], "wait": true/false}
"""

NAVIGATOR_TOOL_DESC = """
2. navigate_to: Automatically navigate to a position on the map grid
   - Input: {"row": 0-8, "col": 0-9}
"""

if USE_NAVIGATOR:
    SYSTEM_PROMPT += NAVIGATOR_TOOL_DESC

SUMMARY_PROMPT = """Please create a detailed summary of the gameplay so far. Include:
1. Key game events and milestones reached
2. Important decisions made
3. Current objectives or goals
4. Current location and Pokémon team status
5. Any strategies or plans mentioned

Provide the summary as plain text."""


class Agent:
    def __init__(self, rom_path, model_name=None, headless=True, sound=False, 
                 max_history=10, load_state=None, device=None):
        """Initialize the agent with a local vision-language model."""
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialization UI
        print(f"\n{Colors.HEADER}{'='*60}")
        print("  GEMINI PLAYS POKEMON - SYSTEM INITIALIZATION")
        print(f"{'='*60}{Colors.RESET}")
        print(f"  DEVICE: {self.device.upper()}")

        # Load model
        model_name = model_name or MODEL_NAME

        print(f"  MODEL:  {model_name}")
        print("-" * 60)

        # Define the Quantization Config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Local VLM Loading Logic
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    dtype=torch.float16,
                    attn_implementation="sdpa"
                )

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.running = True
        self.message_history = []
        self.max_history = max_history

        if load_state:
            print(f"  STATE:  Loading {load_state}")
            self.emulator.load_state(load_state)
        print("="*60 + "\n")

    def _print_boxed_line(self, label, text, width=56, color=Colors.RESET):
        """Helper to wrap text and maintain white box borders with colored text."""
        print(f"{Colors.WHITE}│ {color}{label.ljust(width)}{Colors.WHITE} │{Colors.RESET}")
        
        # Wrap text logic
        words = text.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                current_line += (word + " ")
            else:
                print(f"{Colors.WHITE}│ {color}{current_line.strip().ljust(width)}{Colors.WHITE} │{Colors.RESET}")
                current_line = word + " "
        if current_line:
            print(f"{Colors.WHITE}│ {color}{current_line.strip().ljust(width)}{Colors.WHITE} │{Colors.RESET}")

    def prepare_messages_for_model(self, messages: List[Dict], new_user_content: List[Dict]) -> List[Dict]:
        """Prepare messages in Qwen format"""
        qwen_messages = []

        # Add system prompt
        qwen_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        })
        qwen_messages.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": "Ready."}]
        })

        # Add history
        for msg in messages:
            clean_content = []
            for item in msg["content"]:
                # Only keep text from old messages
                if item["type"] == "text":
                    clean_content.append(item)
            
            if clean_content:
                qwen_messages.append({
                    "role": msg["role"],
                    "content": clean_content
                })

        # Add the new content
        qwen_messages.append({
            "role": "user",
            "content": new_user_content
        })

        return qwen_messages

    def format_prompt(self, messages: List[Dict]) -> str:
        """Format messages into a prompt using the processor."""
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text

    def generate_response(self, screenshot: Image, memory_info: str, collision_map: str = None) -> Dict[str, Any]:
        """Generate a response from the local model."""
        # Prepare the user content
        user_content = [
            {"type": "text", "text": "\nCurrent game screenshot:"},
            {"type": "image", "image": screenshot},
            {"type": "text", "text": f"\nSUPPLEMENTARY DATA (use only if needed):\n"}
        ]

        if collision_map:
            user_content.append({
                "type": "text", 
                "text": f"\nCollision map (shows walkable areas and obstacles):\n{collision_map}"
            })

        user_content.extend([
            {"type": "text", "text": f"\nGame state information from memory:\n{memory_info}"}
        ])

        # Prepare messages
        messages = self.prepare_messages_for_model(self.message_history, user_content)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process images
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
            )

        # Trim input tokens from generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        # Update message history
        self.message_history.append({
            "role": "user",
            "content": user_content
        })
        self.message_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": output_text}]
        })

        return output_text

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the model's JSON response."""
        try:
            # Try to extract JSON from the response
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            response_json = json.loads(response_text)
            return response_json
        except json.JSONDecodeError as e:
            # Return a default action
            return {
                "reasoning": "Failed to parse response, pressing A button",
                "tool_calls": [{"name": "press_buttons", "input": {"buttons": ["a"], "wait": True}}]
            }

    def process_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Process a single tool call."""
        tool_name = tool_call.get("name")
        tool_input = tool_call.get("input", {})

        if tool_name == "press_buttons":
            buttons = tool_input.get("buttons", [])
            wait = tool_input.get("wait", True)

            result = self.emulator.press_buttons(buttons, wait)
            return f"Pressed buttons: {', '.join(buttons)}"

        elif tool_name == "navigate_to":
            row = tool_input.get("row")
            col = tool_input.get("col")

            status, path = self.emulator.find_path(row, col)
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"

            return result
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def run(self, num_steps=1):
        """Main agent loop with non-blocking emulator and clean UI."""
        print(f"{Colors.WHITE}STARTING GAMEPLAY LOOP: {num_steps} STEPS TOTAL{Colors.RESET}\n")     
        BOX_WIDTH = 56 

        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                # Get current game state
                screenshot = self.emulator.get_screenshot()
                memory_info = self.emulator.get_state_from_memory()
                collision_map = self.emulator.get_collision_map()

                screenshot.save("logs/last_vision_input.png")

                # Parse Location and Money for the header
                location = "Unknown"
                money = "$0"
                for line in memory_info.splitlines():
                    if "Location:" in line: location = line.split(":")[1].strip()
                    if "Money:" in line: money = line.split(":")[1].strip()

                # UI: STEP HEADER
                header = f"STEP {steps_completed + 1}/{num_steps} | {location} | {money}"
                print(f"{Colors.WHITE}┌{'─'*(BOX_WIDTH + 2)}┐")
                print(f"│ {Colors.HEADER}{header.ljust(BOX_WIDTH)}{Colors.WHITE} │")
                print(f"├{'─'*(BOX_WIDTH + 2)}┤{Colors.RESET}")

                # AI inference thread logic
                ai_response = {}
                def get_ai_decision():
                    try:
                        response_text = self.generate_response(screenshot, memory_info, collision_map)
                        ai_response['text'] = response_text
                    except Exception as e:
                        ai_response['error'] = str(e)

                ai_thread = threading.Thread(target=get_ai_decision)
                ai_thread.start()

                # Keep emulator running while AI thinks
                last_time = time.perf_counter()
                FRAME_TIME = 1/60 
                while ai_thread.is_alive():
                    current_time = time.perf_counter()
                    delta = current_time - last_time
                    if delta >= FRAME_TIME:
                        self.emulator.pyboy.tick()
                        last_time = current_time - (delta % FRAME_TIME)
                    else:
                        time.sleep(0.001)

                if 'error' in ai_response:
                    print(f"{Colors.WHITE}│ {Colors.ERROR}ERROR: {ai_response['error'][:50].ljust(BOX_WIDTH)}{Colors.WHITE} │")
                    print(f"└{'─'*(BOX_WIDTH + 2)}┘{Colors.RESET}")
                    steps_completed += 1
                    continue

                # Parse the response
                response_json = self.parse_response(ai_response['text'])

                # UI: AI VISION
                visual = response_json.get("visual_observation", "No observation.")
                self._print_boxed_line("AI VISION:", visual, width=BOX_WIDTH, color=Colors.VISION)

                print(f"{Colors.WHITE}│ {' ' * BOX_WIDTH} │{Colors.RESET}")

                # UI: AI THOUGHTS
                reasoning = response_json.get("reasoning", "No reasoning.")
                self._print_boxed_line("AI THOUGHTS:", reasoning, width=BOX_WIDTH, color=Colors.THOUGHTS)

                # UI: ACTION
                tool_calls = response_json.get("tool_calls", [])
                if tool_calls:
                    print(f"{Colors.WHITE}├{'─'*(BOX_WIDTH + 2)}┤{Colors.RESET}")
                    for tool_call in tool_calls:
                        result = self.process_tool_call(tool_call)
                        action_str = f"ACTION: {result}"
                        print(f"{Colors.WHITE}│ {Colors.ACTION}{action_str.ljust(BOX_WIDTH)}{Colors.WHITE} │{Colors.RESET}")
                
                print(f"{Colors.WHITE}└{'─'*(BOX_WIDTH + 2)}┘{Colors.RESET}\n")

                if len(self.message_history) >= self.max_history * 2:
                    self.summarize_history()

                steps_completed += 1

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                import traceback
                traceback.print_exc()
                steps_completed += 1

        if not self.running:
            self.emulator.stop()

        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""

        # Get current state for summary
        screenshot = self.emulator.get_screenshot()
        memory_info = self.emulator.get_state_from_memory()

        # Prepare messages for summarization
        summary_messages = self.message_history.copy()
        summary_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": SUMMARY_PROMPT}
            ]
        })

        # Apply chat template
        text = self.processor.apply_chat_template(
            summary_messages, tokenize=False, add_generation_prompt=True
        )

        # Process
        image_inputs, video_inputs = process_vision_info(summary_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate summary
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        summary_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]


        # Replace message history with summary
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"CONVERSATION HISTORY SUMMARY: {summary_text}"},
                    {"type": "text", "text": f"\nCurrent game state:\n{memory_info}"},
                    {"type": "text", "text": "\nCurrent screenshot:"},
                    {"type": "image", "image": screenshot}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I understand the summary. I'll continue playing from this point."}]
            }
        ]

    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()
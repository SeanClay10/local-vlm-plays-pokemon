import torch
from typing import List, Dict, Any
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import threading
from agent.emulator import Emulator
from transformers import BitsAndBytesConfig
from PIL import Image
import json

class Colors:
    WHITE = '\033[97m'
    HEADER = '\033[96m'
    VISION = '\033[92m'
    THOUGHTS = '\033[93m'
    ACTION = '\033[95m'
    ERROR = '\033[91m'
    RESET = '\033[0m'

class LocalSimpleAgent:
    def __init__(self, rom_path, model_name="Qwen/Qwen3-VL-2B-Instruct", headless=True):
        """Initialize agent with quantized model."""
        self.emulator = Emulator(rom_path, headless, sound=False)
        self.emulator.initialize()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading quantized model on {self.device}...")
        
        # === QUANTIZATION ADDED ===
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="sdpa"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.message_history = []
        self.running = True

    def prepare_messages_for_model(self, messages: List[Dict], new_user_content: List[Dict]) -> List[Dict]:
        """Prepare messages - strip old images to save memory."""
        qwen_messages = []
        
        qwen_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": """You are playing Pokemon Red. Respond in JSON:
            {
                "visual_observation": "What you SEE on screen",
                "reasoning": "Why you're taking this action",
                "tool_calls": [{"name": "press_buttons", "input": {"buttons": ["a"], "wait": true}}]
            }"""}]
        })
        qwen_messages.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": "Ready."}]
        })
        
        for msg in messages:
            clean_content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    clean_content.append(item)
            
            if clean_content:
                qwen_messages.append({
                    "role": msg["role"],
                    "content": clean_content
                })
        
        qwen_messages.append({"role": "user", "content": new_user_content})
        
        return qwen_messages

    def generate_response(self, screenshot: Image, memory_info: str) -> str:
        """Generate response from model (BLOCKING - no threading yet)."""
        user_content = [
            {"type": "text", "text": "Current screenshot:"},
            {"type": "image", "image": screenshot},
            {"type": "text", "text": f"\nGame state:\n{memory_info}"}
        ]
        
        messages = self.prepare_messages_for_model(self.message_history, user_content)
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        # Update history
        self.message_history.append({"role": "user", "content": user_content})
        self.message_history.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
        
        return output_text

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response."""
        try:
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            return json.loads(response_text)
        except:
            return {"reasoning": "Parse error", "tool_calls": [{"name": "press_buttons", "input": {"buttons": ["a"], "wait": True}}]}

    def process_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute tool call."""
        tool_name = tool_call.get("name")
        tool_input = tool_call.get("input", {})
        
        if tool_name == "press_buttons":
            buttons = tool_input.get("buttons", [])
            wait = tool_input.get("wait", True)
            self.emulator.press_buttons(buttons, wait)
            return f"Pressed: {', '.join(buttons)}"
        
        return "Unknown tool"
    
    def run(self, num_steps=10):
        """Main loop with THREADING - emulator stays responsive."""
        print(f"Starting {num_steps} steps...")
        
        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                screenshot = self.emulator.get_screenshot()
                memory_info = self.emulator.get_state_from_memory()
                
                 print(f"\n{Colors.HEADER}Step {steps_completed + 1}/{num_steps}{Colors.RESET}")
                
                ai_response = {}
                
                def get_ai_decision():
                    """Run AI in separate thread."""
                    try:
                        response_text = self.generate_response(screenshot, memory_info)
                        ai_response['text'] = response_text
                    except Exception as e:
                        ai_response['error'] = str(e)
                
                # Start AI thread
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
                
                # Process response
                if 'error' in ai_response:
                    print(f"AI Error: {ai_response['error']}")
                    steps_completed += 1
                    continue
                
                response_json = self.parse_response(ai_response['text'])

                visual = response_json.get("visual_observation", "")
                if visual:
                    print(f"{Colors.VISION}Vision: {visual}{Colors.RESET}")
                
                reasoning = response_json.get("reasoning", "")
                print(f"{Colors.THOUGHTS}Reasoning: {reasoning}{Colors.RESET}")
                
                tool_calls = response_json.get("tool_calls", [])
                for tool_call in tool_calls:
                    result = self.process_tool_call(tool_call)
                    print(f"{Colors.ACTION}Action: {result}{Colors.RESET}")
                
                steps_completed += 1
                if len(self.message_history) >= 20:
                    self.summarize_history()
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"Error: {e}")
                steps_completed += 1
        
        self.stop()
        return steps_completed
    
    def summarize_history(self):
        """Summarize conversation to manage context."""
        print("Summarizing history...")
        
        screenshot = self.emulator.get_screenshot()
        memory_info = self.emulator.get_state_from_memory()
        
        summary_messages = self.message_history.copy()
        summary_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Summarize gameplay progress in 2-3 sentences."}]
        })
        
        text = self.processor.apply_chat_template(summary_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(summary_messages)
        
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=200, temperature=0.7)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        summary_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        # Replace history with summary
        self.message_history = [
            {"role": "user", "content": [
                {"type": "text", "text": f"Summary: {summary_text}\nCurrent state: {memory_info}"},
                {"type": "text", "text": "Current screenshot:"},
                {"type": "image", "image": screenshot}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "Understood."}]}
        ]

    def stop(self):
        """Stop agent."""
        self.running = False
        self.emulator.stop()
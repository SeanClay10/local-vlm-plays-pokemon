import torch
from typing import List, Dict, Any
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from agent.emulator import Emulator
from PIL import Image
import json

class LocalSimpleAgent:
    def __init__(self, rom_path, model_name="Qwen/Qwen2-VL-2B-Instruct", headless=True):
        """Initialize agent with basic model loading."""
        self.emulator = Emulator(rom_path, headless, sound=False)
        self.emulator.initialize()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...")
        
        # Basic model loading (NO quantization yet)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.message_history = []
        self.running = True

    def prepare_messages_for_model(self, messages: List[Dict], new_user_content: List[Dict]) -> List[Dict]:
        """Prepare messages in Qwen format."""
        qwen_messages = []
        
        # Add system prompt
        qwen_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "You are playing Pokemon Red. Respond in JSON with: {\"reasoning\": \"...\", \"tool_calls\": [{\"name\": \"press_buttons\", \"input\": {\"buttons\": [\"a\"], \"wait\": true}}]}"}]
        })
        qwen_messages.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": "Ready."}]
        })
        
        # Add previous messages
        for msg in messages:
            qwen_messages.append(msg)
        
        # Add new content
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

    def stop(self):
        """Stop agent."""
        self.running = False
        self.emulator.stop()
#!/usr/bin/env python3
"""
Ollama-based turtlesim agent.

Subscribes to `/prompt` for natural-language commands. Uses a local Ollama
server to interpret instructions. The model may request tools by returning a
JSON object like {"tool":"move_forward","args":{"distance":2}}. The
node executes the tool and returns the tool result to the model for a final
answer.

Default Ollama server: http://localhost:11434
Default model: llama3.1:8b
"""
import os
import json
import re
import math
import requests
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv


class ROS2AIAgentTurtlesimOllama(Node):
    def __init__(self):
        super().__init__('ros2_ai_agent_turtlesim_ollama')
        self.get_logger().info('ROS2 AI Agent (Ollama) for turtlesim started')

        # Load optional env file
        try:
            share_dir = get_package_share_directory('ros2_basic_agent')
            env_path = Path(share_dir) / 'config' / 'openai.env'
            if env_path.exists():
                load_dotenv(env_path)
        except Exception:
            pass

        # Ollama config
        self.ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        self.model = os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')

        # Turtlesim state
        self.turtle_pose = Pose()
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)

        # Subscribe to prompts
        self.subscription = self.create_subscription(String, 'prompt', self.prompt_callback, 10)

    def call_ollama(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        url = f"{self.ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            text = resp.text

            # Try single JSON
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ('response', 'output', 'text', 'result'):
                        if key in data and isinstance(data[key], str):
                            return data[key]
                    return json.dumps(data)
            except ValueError:
                pass

            assembled = ''
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith('data: '):
                    line = line[len('data: '):]
                try:
                    obj = json.loads(line)
                except ValueError:
                    assembled += line
                    continue
                if isinstance(obj, dict):
                    if 'response' in obj and isinstance(obj['response'], str):
                        assembled += obj['response']
                    elif 'output' in obj and isinstance(obj['output'], str):
                        assembled += obj['output']
                    elif 'text' in obj and isinstance(obj['text'], str):
                        assembled += obj['text']
                    else:
                        assembled += json.dumps(obj)

            # unescape if assembly is quoted JSON
            try:
                unescaped = json.loads(assembled)
                if isinstance(unescaped, str):
                    assembled = unescaped
                else:
                    assembled = json.dumps(unescaped)
            except Exception:
                pass

            return assembled
        except Exception as e:
            self.get_logger().error(f'Error calling Ollama: {e}')
            return f'Error calling Ollama: {e}'

    def pose_callback(self, msg: Pose):
        self.turtle_pose = msg

    # Tool implementations
    def tool_move_forward(self, distance: float) -> str:
        try:
            d = float(distance)
        except Exception:
            return 'Invalid distance'
        # simple proportional velocity for duration
        msg = Twist()
        msg.linear.x = 1.0
        # publish for a short time proportional to distance
        self.cmd_vel_pub.publish(msg)
        self.create_timer(max(0.1, d * 0.5), lambda: self.cmd_vel_pub.publish(Twist()))
        return f'Moved forward {d} units'

    def tool_rotate(self, angle: float) -> str:
        try:
            ang = float(angle)
        except Exception:
            return 'Invalid angle'
        msg = Twist()
        msg.angular.z = math.radians(ang)
        self.cmd_vel_pub.publish(msg)
        self.create_timer(1.0, lambda: self.cmd_vel_pub.publish(Twist()))
        return f'Rotated {ang} degrees'

    def tool_get_pose(self) -> str:
        return f'x: {self.turtle_pose.x:.2f}, y: {self.turtle_pose.y:.2f}, theta: {math.degrees(self.turtle_pose.theta):.2f} deg'

    def extract_json_object(self, text: str):
        """Find the first JSON object in text and return parsed dict or None."""
        # naive search for {...}
        brace_stack = []
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if start is None:
                    start = i
                brace_stack.append(ch)
            elif ch == '}' and brace_stack:
                brace_stack.pop()
                if not brace_stack and start is not None:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        # continue searching
                        start = None
        return None

    def prompt_callback(self, msg: String):
        user_text = msg.data
        system_instructions = (
            "You are a turtlesim assistant. Available tools: move_forward(distance), rotate(angle), get_pose()."
            " If you need a tool to answer, return ONLY a JSON object like {\"tool\":\"get_pose\",\"args\":{}}"
        )

        prompt = system_instructions + "\n\nUser: " + user_text + "\n\nAnswer:"
        self.get_logger().info('Sending prompt to Ollama')
        first_reply = self.call_ollama(prompt)
        self.get_logger().info(f'First model reply: {first_reply}')

        # Try robust extraction of a JSON tool call
        tool_call = None
        try:
            parsed = json.loads(first_reply)
            if isinstance(parsed, dict) and 'tool' in parsed:
                tool_call = parsed
        except Exception:
            # try to find a JSON object inside text
            tool_call = self.extract_json_object(first_reply)

        final_answer = None
        if tool_call:
            tool_name = tool_call.get('tool')
            args = tool_call.get('args', {}) or {}
            if tool_name == 'move_forward':
                tool_result = self.tool_move_forward(args.get('distance', 1))
            elif tool_name == 'rotate':
                tool_result = self.tool_rotate(args.get('angle', 90))
            elif tool_name == 'get_pose':
                tool_result = self.tool_get_pose()
            else:
                tool_result = f'Unknown tool: {tool_name}'

            self.get_logger().info(f'Executed tool {tool_name}: {tool_result}')

            followup = (
                system_instructions
                + "\n\nUser: " + user_text
                + "\n\nTool call: " + json.dumps(tool_call)
                + "\nTool result: " + tool_result
                + "\n\nNow provide a final concise answer."
            )

            final_answer = self.call_ollama(followup)
            self.get_logger().info(f'Final model reply: {final_answer}')
        else:
            final_answer = first_reply

        # Log and optionally publish the final answer
        try:
            self.get_logger().info(f'Result: {final_answer}')
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ROS2AIAgentTurtlesimOllama()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Ollama-based TurtleBot3 Nav2 agent.

Subscribes to `/prompt` for natural-language commands. Uses a local Ollama
server to interpret instructions. The model may request tools by returning a
JSON object like {"tool":"move_to_goal","args":{"x":1.0,"y":2.0}}. The
node executes the tool and returns the tool result to the model for a final
answer.

Default Ollama server: http://192.168.5.70:11434
Default model: llama3.1:8b
"""
import os
import json
import math
import requests
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from transforms3d.euler import quat2euler


class ROS2AIAgentTurtlebot3Nav2Ollama(Node):
    def __init__(self):
        super().__init__('ros2_ai_agent_turtlebot3_nav2_ollama')
        self.get_logger().info('ROS2 AI Agent (Ollama) for TurtleBot3 Nav2 started')

        # State
        self.current_pose = PoseStamped()

        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subscriber for odometry
        self.pose_sub = self.create_subscription(Odometry, '/odom', self.pose_callback, 10)

        # Load optional env
        try:
            share_dir = get_package_share_directory('ros2_basic_agent')
            env_path = Path(share_dir) / 'config' / 'openai.env'
            if env_path.exists():
                load_dotenv(env_path)
        except Exception:
            pass

        # Ollama config
        self.ollama_url = os.environ.get('OLLAMA_URL', 'http://192.168.5.70:11434')
        self.model = os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')

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

            # single JSON
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

            # unescape potential quoted JSON
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

    def pose_callback(self, msg: Odometry):
        self.current_pose.pose = msg.pose.pose
        self.current_pose.header = msg.header

    def move_to_goal(self, x: float, y: float) -> str:
        try:
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = float(x)
            goal_msg.pose.pose.position.y = float(y)
            goal_msg.pose.pose.orientation.w = 1.0

            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                return 'Navigation server not available'

            self.nav_client.send_goal_async(goal_msg)
            return f'Navigating to position x: {x}, y: {y}'
        except Exception as e:
            return f'Error sending goal: {e}'

    def get_current_pose(self) -> str:
        try:
            x = self.current_pose.pose.position.x
            y = self.current_pose.pose.position.y
            orientation = self.current_pose.pose.orientation
            roll, pitch, yaw = quat2euler([
                orientation.w,
                orientation.x,
                orientation.y,
                orientation.z
            ])
            return f'x: {x:.2f}, y: {y:.2f}, theta: {math.degrees(yaw):.2f} degrees'
        except Exception as e:
            return f'Error getting pose: {e}'

    def extract_json_object(self, text: str):
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
                        start = None
        return None

    def prompt_callback(self, msg: String):
        user_text = msg.data
        system_instructions = (
            'You are a TurtleBot3 Nav2 assistant. Available tools: move_to_goal(x,y), get_current_pose(). '
            'If you need a tool to answer, return ONLY a JSON object like {"tool":"move_to_goal","args":{"x":1.0,"y":2.0}}'
        )

        prompt = system_instructions + '\n\nUser: ' + user_text + '\n\nAnswer:'
        self.get_logger().info('Sending prompt to Ollama')
        first_reply = self.call_ollama(prompt)
        self.get_logger().info(f'First model reply: {first_reply}')

        tool_call = None
        try:
            parsed = json.loads(first_reply)
            if isinstance(parsed, dict) and 'tool' in parsed:
                tool_call = parsed
        except Exception:
            tool_call = self.extract_json_object(first_reply)

        final_answer = None
        if tool_call:
            tool_name = tool_call.get('tool')
            args = tool_call.get('args', {}) or {}
            if tool_name == 'move_to_goal':
                tool_result = self.move_to_goal(args.get('x', 0), args.get('y', 0))
            elif tool_name == 'get_current_pose':
                tool_result = self.get_current_pose()
            else:
                tool_result = f'Unknown tool: {tool_name}'

            self.get_logger().info(f'Executed tool {tool_name}: {tool_result}')

            followup = (
                system_instructions
                + '\n\nUser: ' + user_text
                + '\n\nTool call: ' + json.dumps(tool_call)
                + '\nTool result: ' + tool_result
                + '\n\nNow provide a final concise answer.'
            )

            final_answer = self.call_ollama(followup)
            self.get_logger().info(f'Final model reply: {final_answer}')
        else:
            final_answer = first_reply

        try:
            self.get_logger().info(f'Result: {final_answer}')
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ROS2AIAgentTurtlebot3Nav2Ollama()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()

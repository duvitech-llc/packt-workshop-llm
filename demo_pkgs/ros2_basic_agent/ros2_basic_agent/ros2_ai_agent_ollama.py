#!/usr/bin/env python3
"""
ROS2 node that uses a local Ollama LLM to answer prompts and run a couple of simple tools.

Behavior:
- Subscribes to the `prompt` topic (std_msgs/String).
- Sends the incoming prompt to a local Ollama server and expects either a plain-text answer
  or a JSON object indicating a tool call, e.g. {"tool":"get_ros_distro","args":{}}.
- If the model requests a tool, the node executes it and provides the tool result back to
  the model in a second request so the model can return a final answer.

Configuration:
- By default this will talk to Ollama at `http://localhost:11434` and use
  model `llama3.1:8b`.
- These values can be overridden with environment variables `OLLAMA_URL` and `OLLAMA_MODEL`.

This implementation intentionally keeps dependencies minimal (uses `requests`).
"""
import os
import json
import re
import requests
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv


class ROS2AIAgentOllama(Node):
    def __init__(self):
        super().__init__('ros2_ai_agent_ollama')
        self.get_logger().info('ROS2 AI Agent (Ollama) started')

        # Load package config if present (keeps parity with the other agent)
        try:
            share_dir = get_package_share_directory('ros2_basic_agent')
            env_path = Path(share_dir) / 'config' / 'openai.env'
            if env_path.exists():
                load_dotenv(env_path)
        except Exception:
            # not fatal, continue
            pass

        # Ollama server URL and model
        self.ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        self.model = os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')

        # Subscribe to prompts
        self.subscription = self.create_subscription(
            String,
            'prompt',
            self.prompt_callback,
            10,
        )

    def call_ollama(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """Call the Ollama local HTTP API and return the model text output.

        This function uses the simple `/api/generate` POST interface which many
        Ollama deployments provide. If your Ollama API uses a different shape
        adapt the payload accordingly.
        """
        url = f"{self.ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            # send the raw prompt; many Ollama builds accept `prompt` as a string
            "prompt": prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()

            text = resp.text

            # If the response is a single valid JSON object, prefer that.
            try:
                data = json.loads(text)
                # Common single-response shapes
                if isinstance(data, dict):
                    for key in ('response', 'output', 'text', 'result'):
                        if key in data and isinstance(data[key], str):
                            return data[key]
                    # fallback to stringified JSON
                    return json.dumps(data)
            except ValueError:
                # Not a single JSON object — might be newline-delimited JSON (streaming).
                pass

            # Try to parse newline-delimited JSON chunks and assemble 'response' fields
            assembled = ''
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Some servers prefix with 'data: ' like SSE; remove that if present
                if line.startswith('data: '):
                    line = line[len('data: '):]
                try:
                    obj = json.loads(line)
                except ValueError:
                    # Not JSON: append raw line
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
                        # if the dict itself looks like the final payload, stringify
                        assembled += json.dumps(obj)

            # If assembled contains an escaped JSON string like "{\"tool\":...}\"",
            # try to unescape it to real JSON/text.
            try:
                unescaped = json.loads(assembled)
                if isinstance(unescaped, str):
                    assembled = unescaped
                else:
                    assembled = json.dumps(unescaped)
            except Exception:
                # If unescape fails, keep assembled as-is
                pass

            return assembled
        except Exception as e:
            self.get_logger().error(f'Error calling Ollama: {e}')
            return f'Error calling Ollama: {e}'

    def get_ros_distro(self) -> str:
        """Get the current ROS distribution name from environment."""
        try:
            ros_distro = os.environ.get('ROS_DISTRO')
            if ros_distro:
                return f"Current ROS distribution: {ros_distro}"
            return "ROS distribution environment variable (ROS_DISTRO) not set"
        except Exception as e:
            return f"Error getting ROS distribution: {e}"

    def get_domain_id(self) -> str:
        """Get the current ROS_DOMAIN_ID from environment."""
        try:
            domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
            return f"Current ROS domain ID: {domain_id}"
        except Exception as e:
            return f"Error getting ROS domain ID: {e}"

    def prompt_callback(self, msg: String):
        """Main handler for incoming prompts.

        The flow:
        1. Send prompt to model with an instruction that it may request a tool call by
           returning a JSON object: {"tool":"<name>","args":{}}.
        2. If model responds with a tool call, execute the tool and give the tool result
           back to the model in a second request so it can produce a final answer.
        3. If model responds with plain text, treat it as the final answer.
        """
        user_text = msg.data

        system_instructions = (
            "You are a ROS 2 system information assistant. "
            "Available tools: get_ros_distro(), get_domain_id().\n"
            "If you need to run a tool to answer, return a single JSON object ONLY,"
            " for example: {\"tool\":\"get_ros_distro\",\"args\":{}}."
            " If no tool is needed, return the final answer as plain text."
        )

        prompt = system_instructions + "\n\nUser: " + user_text + "\n\nAnswer:"

        self.get_logger().info('Sending prompt to Ollama')
        first_reply = self.call_ollama(prompt)
        self.get_logger().info(f'First model reply: {first_reply}')

        # Try to interpret a JSON tool call
        tool_call = None
        try:
            parsed = json.loads(first_reply)
            if isinstance(parsed, dict) and 'tool' in parsed:
                tool_call = parsed
        except Exception:
            tool_call = None

        final_answer = None

        if tool_call:
            tool_name = tool_call.get('tool')
            # Only two allowed tools here
            if tool_name == 'get_ros_distro':
                tool_result = self.get_ros_distro()
            elif tool_name == 'get_domain_id':
                tool_result = self.get_domain_id()
            else:
                tool_result = f"Unknown tool requested: {tool_name}"

            self.get_logger().info(f"Executed tool {tool_name}: {tool_result}")

            # Give the model the result and ask for a final answer
            followup = (
                system_instructions
                + "\n\nUser: "
                + user_text
                + "\n\nTool call: "
                + json.dumps(tool_call)
                + "\nTool result: "
                + tool_result
                + "\n\nNow provide a final concise answer to the user."
            )

            self.get_logger().info('Sending tool result back to Ollama for final answer')
            final_answer = self.call_ollama(followup)
            self.get_logger().info(f'Final model reply: {final_answer}')
        else:
            # Model returned plain text; treat as final answer.
            final_answer = first_reply

        # Log the answer (mirrors behavior of the existing basic agent)
        try:
            self.get_logger().info(f"Result: {final_answer}")
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ROS2AIAgentOllama()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()

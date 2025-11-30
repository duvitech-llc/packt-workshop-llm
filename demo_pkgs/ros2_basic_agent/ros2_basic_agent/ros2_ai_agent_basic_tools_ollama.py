#!/usr/bin/env python3
"""
ROS2 node that uses Ollama (via LangChain) as the LLM for an agent that can call ROS2 inspection tools.
"""

import os
import math
import subprocess
from pathlib import Path
from typing import List

from std_msgs.msg import String
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from transforms3d.euler import quat2euler

from dotenv import load_dotenv

# LangChain imports
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_agent  # general agent factory
# Ollama integration for LangChain (install langchain-ollama or langchain_ollama)
try:
    # Newer dedicated package
    from langchain_ollama import ChatOllama  # recommended binding
except Exception:
    # Fallback to the older names sometimes used in different setups
    try:
        from langchain.llms.ollama import Ollama as ChatOllama
    except Exception:
        ChatOllama = None  # We'll error later if not installed


class ROS2AIAgent(Node):
    def __init__(self):
        super().__init__('ros2_ai_agent')
        self.get_logger().info('ROS2 AI Agent has been started')

        # Tools (wrapped as langchain tools)
        self.list_topics_tool = tool(self.list_topics)
        self.list_nodes_tool = tool(self.list_nodes)
        self.list_services_tool = tool(self.list_services)
        self.list_actions_tool = tool(self.list_actions)

        # Prompt guiding the agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a ROS 2 system information assistant.
            You can check ROS 2 system status using these commands:
            - list_topics(): List all available ROS 2 topics
            - list_nodes(): List all running ROS 2 nodes
            - list_services(): List all available ROS 2 services
            - list_actions(): List all available ROS 2 actions

            Return only the necessary information and results. e.g
            Human: Show me all running nodes
            AI: Here are the running ROS 2 nodes: [node list]
            """),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        # load env (if you have any config for Ollama client)
        share_dir = get_package_share_directory('ros2_basic_agent')
        config_dir = share_dir + '/config' + '/openai.env'  # you may rename this if needed
        if Path(config_dir).exists():
            load_dotenv(Path(config_dir))

        # toolkit = list of tool functions exposed to the agent
        self.toolkit = [
            self.list_topics_tool,
            self.list_nodes_tool,
            self.list_services_tool,
            self.list_actions_tool
        ]

        # ---- LLM setup: Ollama ----
        if ChatOllama is None:
            self.get_logger().error(
                "ChatOllama / langchain_ollama not available. "
                "Install langchain-ollama and ollama client (see docs)."
            )
            raise RuntimeError("langchain_ollama not installed or import failed")

        # Example: use a local model you pulled into Ollama, e.g. 'llama3.1' or 'gpt-oss:20b'
        # Tune model name and temperature as you like
        self.llm = ChatOllama(model="llama3.1", temperature=0)

        # ---- Agent construction ----
        # Use LangChain's generic create_agent so the same agent logic can run with Ollama
        # create_agent signatures evolve across langchain versions; this is a reasonable pattern.
        self.agent = create_agent(self.llm, tools=self.toolkit, prompt=self.prompt)

        # Wrap the agent in an executor (controls running; consistent with earlier pattern)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.toolkit, verbose=True)

        # ROS2 subscription
        self.subscription = self.create_subscription(
            String,
            'prompt',
            self.prompt_callback,
            10
        )

    # Tool implementations
    def list_topics(self) -> str:
        """List all available ROS 2 topics."""
        try:
            result = subprocess.run(['ros2', 'topic', 'list'],
                                     capture_output=True, text=True, check=True)
            return f"Available ROS 2 topics:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error listing topics: {str(e)}"

    def list_nodes(self) -> str:
        """List all running ROS 2 nodes."""
        try:
            result = subprocess.run(['ros2', 'node', 'list'],
                                     capture_output=True, text=True, check=True)
            return f"Running ROS 2 nodes:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error listing nodes: {str(e)}"

    def list_services(self) -> str:
        """List all available ROS 2 services."""
        try:
            result = subprocess.run(['ros2', 'service', 'list'],
                                     capture_output=True, text=True, check=True)
            return f"Available ROS 2 services:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error listing services: {str(e)}"

    def list_actions(self) -> str:
        """List all available ROS 2 actions."""
        try:
            result = subprocess.run(['ros2', 'action', 'list'],
                                     capture_output=True, text=True, check=True)
            return f"Available ROS 2 actions:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error listing actions: {str(e)}"

    def prompt_callback(self, msg):
        try:
            # AgentExecutor expects same input as the prompt variables
            result = self.agent_executor.invoke({"input": msg.data})
            # The structure of AgentExecutor result can vary by LangChain version.
            # Typically it's {"output": "<final response>"} or AgentFinish-like structure.
            out = None
            if isinstance(result, dict):
                out = result.get("output") or result.get("text") or str(result)
            else:
                out = str(result)
            self.get_logger().info(f"Result: {out}")
        except Exception as e:
            self.get_logger().error(f'Error processing prompt: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ROS2AIAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()

# GF AgentKit

This is a work-in-progress. I'll add to it as I go.

Simple agent kit that uses Pydantic AI framework, with ready-to-go soft tools and context management.

One of the issues with using tools is their "gravity" over the context.
Maybe providers that provide passing tools on prompt, create a hard bias toward using the tools (a.k.a hard-tools). As this might be good for workflow-oriented agent systems, it doesn't fit well to reasoning-oriented flows, due to the strong bias toward tool-using.

This small library provides ready-to-use soft-tool agents, built on Pydantic AI, with some ready to use tools.

## Agents
DelegatingToolsAgent - expose a single tool to the agent (lowers the gravity of multiple tool). This one tool is delegating to a tool executing agent that holds all the tools. The executing agent can also provide as an additional logic layer before and after tool calling.


## Install
`pip install agentkit-gf`
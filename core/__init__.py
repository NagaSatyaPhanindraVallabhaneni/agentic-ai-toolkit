"""Core agentic AI framework: tool protocol, agent step tracking, and a
ReAct-style (Thought -> Action -> Observation) orchestrator loop.

Everything here is dependency-light and LLM-agnostic by design. The
"brain" of any agent built on this framework is an `LLMPolicyProtocol`
implementation — this package ships deterministic, dependency-free
policies for the demo projects, but a production deployment swaps in a
real model call (Claude, GPT-4.1, etc.) without touching the orchestrator
or tool-calling machinery at all.
"""

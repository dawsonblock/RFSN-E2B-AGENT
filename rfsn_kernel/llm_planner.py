"""LLM Planner - Concrete planner using language models.

This planner wraps LLM calls and converts outputs to proposals.
It is ISOLATED from direct system access.

INVARIANTS:
1. Planner only produces proposals (never executes)
2. LLM responses are parsed into structured proposals
3. Invalid LLM responses are rejected (not executed)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .gate import GateDecision
from .planner_interface import Planner
from .proposal import Proposal, ProposalIntent, create_proposal
from .state import StateSnapshot

logger = logging.getLogger(__name__)


@dataclass
class LLMPlannerConfig:
    """Configuration for LLM Planner."""
    
    # Model settings
    provider: str = "deepseek"  # "deepseek", "openai", "anthropic"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 4096
    
    # API settings
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str | None = None
    
    # Behavior
    max_retries: int = 3
    include_rejection_feedback: bool = True
    max_context_files: int = 5
    
    # Prompt settings
    system_prompt: str = ""
    prompt_variant: str = "v_minimal_fix"


# System prompt template
SYSTEM_PROMPT_TEMPLATE = """You are an expert software engineer working on a code repair task.

Your role is to propose ONE action at a time. Each proposal must be structured as JSON.

CONSTRAINTS:
- You can ONLY propose actions, not execute them
- Your proposals will be validated by a safety gate
- Invalid proposals will be rejected with feedback

ALLOWED ACTIONS:
- modify_file: Modify an existing file with a diff patch
- create_file: Create a new file
- run_test: Run tests to verify changes
- read_file: Read a file to understand the codebase
- search_repo: Search for patterns in the repository

RESPONSE FORMAT:
```json
{{
    "intent": "modify_file",
    "target": "path/to/file.py",
    "patch": "--- a/path/to/file.py\\n+++ b/path/to/file.py\\n@@ -10,3 +10,4 @@\\n...",
    "justification": "Why this change is needed",
    "expected_effect": "What this change should accomplish"
}}
```

Current task: {task}
Current state: Step {step_index}, Tests: {test_status}
"""


def get_llm_client(config: LLMPlannerConfig):
    """Get appropriate LLM client based on provider.
    
    Returns:
        Client with chat.completions.create() method.
    """
    try:
        if config.provider == "deepseek":
            from openai import OpenAI
            api_key = os.environ.get(config.api_key_env) or os.environ.get("DEEPSEEK_API_KEY")
            return OpenAI(
                api_key=api_key,
                base_url=config.base_url or "https://api.deepseek.com",
            )
        elif config.provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get(config.api_key_env) or os.environ.get("OPENAI_API_KEY")
            return OpenAI(api_key=api_key)
        elif config.provider == "anthropic":
            # Anthropic uses a different client interface
            import anthropic
            api_key = os.environ.get(config.api_key_env) or os.environ.get("ANTHROPIC_API_KEY")
            return anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
    except ImportError as e:
        logger.error(f"Failed to import LLM client: {e}")
        raise


def parse_llm_response(response: str) -> dict[str, Any] | None:
    """Parse LLM response to extract proposal JSON.
    
    Args:
        response: Raw LLM response text.
    
    Returns:
        Parsed proposal dict or None if parsing fails.
    """
    # Try to extract JSON from code block
    json_match = re.search(r"```(?:json)?\s*\n?({.*?})\s*\n?```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object
    json_match = re.search(r"\{[^{}]*\"intent\"[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try entire response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    return None


class LLMPlanner(Planner):
    """LLM-based planner that generates proposals from language model outputs.
    
    INVARIANTS:
    1. Only produces proposals (never executes)
    2. LLM responses are parsed into structured proposals
    3. Invalid responses are rejected
    
    Usage:
        planner = LLMPlanner(LLMPlannerConfig(
            provider="deepseek",
            model="deepseek-chat",
        ))
        
        proposal = planner.propose(state, task)
    """
    
    def __init__(self, config: LLMPlannerConfig | None = None):
        """Initialize LLM planner.
        
        Args:
            config: Planner configuration.
        """
        self.config = config or LLMPlannerConfig()
        self._client = None
        self._rejection_history: list[tuple[Proposal, str]] = []
        self._call_count = 0
    
    @property
    def client(self):
        """Lazy-load LLM client."""
        if self._client is None:
            self._client = get_llm_client(self.config)
        return self._client
    
    def propose(
        self,
        state: StateSnapshot,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Proposal | None:
        """Generate a proposal using the LLM.
        
        Args:
            state: Current state snapshot.
            task: Task description.
            context: Optional context.
        
        Returns:
            Proposal or None if generation fails.
        """
        context = context or {}
        
        # Build system prompt
        system_prompt = self.config.system_prompt or SYSTEM_PROMPT_TEMPLATE.format(
            task=task,
            step_index=state.task_progress.step_index,
            test_status=state.test_result.status.value,
        )
        
        # Build user message with context
        user_message = self._build_user_message(state, task, context)
        
        # Call LLM
        try:
            response = self._call_llm(system_prompt, user_message)
            self._call_count += 1
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
        
        # Parse response
        proposal_data = parse_llm_response(response)
        if proposal_data is None:
            logger.warning("Failed to parse LLM response")
            return None
        
        # Convert to Proposal
        try:
            proposal = create_proposal(
                intent=proposal_data.get("intent", "modify_file"),
                target=proposal_data.get("target", ""),
                justification=proposal_data.get("justification", "LLM generated"),
                expected_effect=proposal_data.get("expected_effect", "Unknown"),
                patch=proposal_data.get("patch", ""),
                test_command=proposal_data.get("test_command", ""),
                search_query=proposal_data.get("search_query", ""),
                confidence=proposal_data.get("confidence", 0.5),
                metadata={"llm_model": self.config.model},
            )
            return proposal
        except Exception as e:
            logger.error(f"Failed to create proposal: {e}")
            return None
    
    def observe_rejection(
        self,
        proposal: Proposal,
        decision: GateDecision,
    ) -> None:
        """Record rejection for future context.
        
        Args:
            proposal: Rejected proposal.
            decision: Gate decision.
        """
        if self.config.include_rejection_feedback:
            self._rejection_history.append((proposal, decision.reason))
            # Limit history size
            if len(self._rejection_history) > 10:
                self._rejection_history = self._rejection_history[-10:]
    
    def _build_user_message(
        self,
        state: StateSnapshot,
        task: str,
        context: dict[str, Any],
    ) -> str:
        """Build user message with state and context."""
        parts = [
            f"## Task\n{task}\n",
            f"## Current State",
            f"- Step: {state.task_progress.step_index}",
            f"- Patches applied: {state.task_progress.patches_applied}",
            f"- Test status: {state.test_result.status.value}",
        ]
        
        # Add rejection feedback
        if self._rejection_history and self.config.include_rejection_feedback:
            parts.append("\n## Recent Rejections (learn from these)")
            for proposal, reason in self._rejection_history[-3:]:
                parts.append(f"- {proposal.intent.value} on {proposal.target}: {reason}")
        
        # Add test output if available
        if state.test_result.stderr:
            parts.append(f"\n## Test Output\n```\n{state.test_result.stderr[:1000]}\n```")
        
        parts.append("\n## Your Proposal\nProvide a single JSON proposal:")
        
        return "\n".join(parts)
    
    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call the LLM and return response text."""
        if self.config.provider == "anthropic":
            # Anthropic has different API
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        else:
            # OpenAI-compatible API
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
    
    def get_name(self) -> str:
        """Get planner name."""
        return f"LLMPlanner({self.config.provider}/{self.config.model})"
    
    def get_config(self) -> dict[str, Any]:
        """Get planner config for audit."""
        return {
            "name": self.get_name(),
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "call_count": self._call_count,
        }


class MockLLMPlanner(Planner):
    """Mock LLM planner for testing without API calls.
    
    Returns predefined responses based on patterns.
    """
    
    def __init__(self, responses: list[dict[str, Any]] | None = None):
        """Initialize with mock responses.
        
        Args:
            responses: List of proposal dicts to return.
        """
        self._responses = responses or []
        self._index = 0
        self._rejections: list[tuple[Proposal, GateDecision]] = []
    
    def propose(
        self,
        state: StateSnapshot,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Proposal | None:
        """Return next mock response."""
        if self._index >= len(self._responses):
            return None
        
        response = self._responses[self._index]
        self._index += 1
        
        try:
            return create_proposal(
                intent=response.get("intent", "modify_file"),
                target=response.get("target", "test.py"),
                justification=response.get("justification", "Mock proposal"),
                expected_effect=response.get("expected_effect", "Test effect"),
                patch=response.get("patch", ""),
            )
        except Exception:
            return None
    
    def observe_rejection(
        self,
        proposal: Proposal,
        decision: GateDecision,
    ) -> None:
        """Record rejection."""
        self._rejections.append((proposal, decision))

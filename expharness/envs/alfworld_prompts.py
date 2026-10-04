"""
ALFWorld prompt templates for ExpHarness.

Step 0: Multi-turn skill search (Copilot, RL-trained)
Agent interaction: Multi-turn environment interaction (frozen executor LLM)
"""

# ============================================================================
# Step 0: Copilot Skill Search Templates
# ============================================================================

STEP0_SEARCH_TEMPLATE = """\
You are a retrieval strategy controller for an embodied agent in the ALFRED household environment.

Task: {task_description}

You control how experiences are retrieved from a knowledge graph by setting two parameters:

R (0-100): Graph exploration scope.
  Low R (0-30): Explore widely through connected experiences in the graph (broad discovery).
  High R (70-100): Stay close to the most similar experiences (precise, local search).

W (0-100): Selection preference between semantic similarity and proven effectiveness.
  Low W (0-30): Prefer experiences most similar to this task (safe, like basic search).
  High W (70-100): Prefer experiences that historically led to task success (trust past results).

When W=0, retrieval is identical to basic cosine search. Increase W to leverage historical feedback.

Output format: <search>R:W</search>
You may reason briefly in <think>...</think> tags first.
"""

STEP0_SEARCH_EXAMPLE = """\
Examples:
- New/unfamiliar task type → <search>20:10</search>
- Common task with good past data → <search>80:70</search>
- Moderate confidence → <search>50:40</search>

Output your retrieval parameters now. Complete the tag:
<search>"""

# ============================================================================
# Agent Interaction Templates (NVIDIA API, frozen)
# ============================================================================

AGENT_FIRST_ACTION_TEMPLATE = """\
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

## Retrieved Experiences

{retrieved_experiences}

Now take an action. Reason briefly, then choose an admissible action.
<think>Brief reasoning</think>
<action>your chosen action</action>
"""

AGENT_FIRST_ACTION_NO_EXP_TEMPLATE = """\
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now take an action. Reason briefly, then choose an admissible action.
<think>Brief reasoning</think>
<action>your chosen action</action>
"""

AGENT_STEP_TEMPLATE = """\
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}

## Retrieved Experiences

{retrieved_experiences}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now take an action. Reason briefly, then choose an admissible action.
<think>Brief reasoning</think>
<action>your chosen action</action>
"""

AGENT_STEP_NO_EXP_TEMPLATE = """\
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now take an action. Reason briefly, then choose an admissible action.
<think>Brief reasoning</think>
<action>your chosen action</action>
"""

# Agent Enhancement Plan

This document outlines how to extend the project so the agent can make use of all
locally installed Ollama models and intelligently pick the best one for a task
while defaulting to `rolandroland/llama3.1-uncensored:latest`.

## 1. Environment Setup

1. Install the latest version of **Ollama** and ensure it is running on the
   machine.
2. Pull or create any models you wish to use with `ollama pull <model>`.
3. Verify available models with `ollama list`.

## 2. Model Discovery and Management

1. During agent startup, call `ollama.list()` to retrieve all locally installed
   models.  Store both the model name and size so they can be displayed or
   selected later.
2. Keep `rolandroland/llama3.1-uncensored:latest` in the list and treat it as the
   primary model.
3. Implement a refresh method to update the list of models without restarting the
   agent.
4. Track model performance (usage count, success rate, response time) in
   `model_performance.json` so future tasks can be routed to the most effective
   models.

## 3. Task Analysis and Model Selection

1. Use `TaskAnalyzer` to classify a user request by domain and complexity.
2. Ask `IntelligentModelManager.get_best_model_for_task` for the optimal model
   based on domain, complexity, and past performance.
3. When the agent encounters a new task, it should:
   - Re-check available models if necessary.
   - Choose the model with the highest score and fall back to the main model when
     no good match exists.
4. Log which model was chosen so the user can inspect model usage later.

## 4. Self-Improvement and Adaptation

1. Record failures in `failure_patterns.json` and use
   `CapabilityGapAnalyzer` together with `DeepCoderInterface` to generate new
   capabilities when a task cannot be completed.
2. Store experience data in `self_model.json` so `MetaCognition` can reflect on
   what was learned and adjust strategies over time.
3. Continuously update the agent's internal plan and performance metrics so it
   becomes better at choosing models and deciding when to generate new code.

## 5. Automation and Vision

1. Web automation is provided by `WebMasterPro` using Selenium and PyAutoGUI for
   screen interactions. Enable `pyautogui` failsafe and allow pixel-level
   operations for tasks that require visual navigation.
2. Collect screenshots in the `screenshots` directory for later analysis.
3. Consider using a vision-capable model (for example `llava`) to interpret
   screenshots when reasoning about what is displayed on the screen.

## 6. Execution Loop

1. In interactive mode the main loop reads user commands. Add a command
   (`list-models`) that prints discovered models with size information so the
   user sees what is available.
2. When processing a request the agent will:
   - Analyze the task.
   - Plan the sequence of actions.
   - Execute the plan using the best model and available capabilities.
   - Debug and retry failed steps automatically.
   - Store execution traces for reflection and testing.
3. After each task, performance metrics should be updated so the next invocation
   can learn from previous runs.

## 7. Testing

1. Keep unit tests under `tests/` for critical components such as task analysis
   and model discovery.  Running `pytest` should verify behaviour after each
   change.
2. Optionally create end-to-end tests simulating tasks across different models to
   ensure the agent can switch models when required.

This plan aims to provide the necessary steps to make the agent aware of all
Ollama models installed on the PC, intelligently use them to complete tasks, and
continuously improve through reflection and new capability generation.

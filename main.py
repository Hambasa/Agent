#!/usr/bin/env python3
"""
ULTIMATE SELF-IMPROVING INTELLIGENT AGENT
This agent can think, learn, plan, execute, debug, adapt, reason, and be self-aware.
It works like Cline - analyzing tasks, breaking them down step by step, and automatically
developing new capabilities when it encounters limitations.
"""

import ollama
import json
import time
import logging
import os
import sys
import subprocess
import webbrowser
import re
import uuid
import importlib
import traceback
import ast
import inspect
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import tempfile

# Custom JSON encoder for Enums and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        return super().default(obj)

# Try to import web automation libraries
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class TaskDomain(Enum):
    CODING = "coding"
    WEB_AUTOMATION = "web_automation"
    REASONING = "reasoning"
    PLANNING = "planning"
    ANALYSIS = "analysis"
    CREATIVITY = "creativity"
    DEBUGGING = "debugging"
    SYSTEM = "system"
    VISION = "vision"
    GENERAL = "general"

class StrategyResult(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    REQUIRES_NEW_CAPABILITY = "requires_new_capability"

@dataclass
class GoalCriteria:
    task_description: str
    success_indicators: List[str]
    evidence_required: List[str]
    validation_methods: List[str]
    failure_indicators: List[str]
    
@dataclass
class StrategyAttempt:
    strategy_name: str
    approach: str
    parameters: Dict[str, Any]
    result: StrategyResult
    execution_details: Dict[str, Any]
    evidence_collected: List[str]
    failure_reason: Optional[str] = None
    execution_time: float = 0.0

@dataclass
class ModelCapability:
    name: str
    size_gb: float
    domains: List[TaskDomain]
    complexity_rating: int  # 1-10
    performance_history: Dict[str, float]
    usage_count: int
    success_rate: float
    average_response_time: float
    specialties: List[str]

@dataclass
class TaskAnalysis:
    description: str
    complexity: TaskComplexity
    domain: TaskDomain
    estimated_steps: int
    required_capabilities: List[str]
    confidence: float
    alternative_approaches: List[str]

@dataclass
class ExecutionStep:
    number: int
    description: str
    action_type: str
    parameters: Dict[str, Any]
    expected_outcome: str
    completed: bool = False
    success: bool = False
    error: Optional[str] = None
    execution_time: float = 0.0
    output: Any = None

class IntelligentModelManager:
    """Manages all AI models and automatically selects the best one for each task"""
    
    def __init__(self, ollama_client):
        self.ollama = ollama_client
        self.models: Dict[str, ModelCapability] = {}
        self.performance_cache = {}
        self.load_performance_data()
        self.discover_and_assess_models()
    
    def discover_and_assess_models(self):
        """Discover all available models and assess their capabilities"""
        try:
            raw_models = self.ollama.list()
            logger.info(f"🔍 Discovering and assessing {len(raw_models.get('models', []))} models...")
            
            # Map model names to their capabilities based on your available models
            model_mappings = {
                'deepcoder': ModelCapability(
                    name='deepcoder:latest',
                    size_gb=9.0,
                    domains=[TaskDomain.CODING, TaskDomain.DEBUGGING, TaskDomain.REASONING, TaskDomain.PLANNING],
                    complexity_rating=10,
                    performance_history={},
                    usage_count=0,
                    success_rate=0.9,
                    average_response_time=3.0,
                    specialties=['self_improvement', 'capability_generation', 'complex_problem_solving']
                ),
                'qwen2.5': ModelCapability(
                    name='hf.co/mradermacher/Qwen2.5-7B-sft-SPIN-gpt4o-GGUF:Q8_0',
                    size_gb=8.1,
                    domains=[TaskDomain.REASONING, TaskDomain.PLANNING, TaskDomain.ANALYSIS],
                    complexity_rating=9,
                    performance_history={},
                    usage_count=0,
                    success_rate=0.85,
                    average_response_time=2.5,
                    specialties=['advanced_reasoning', 'complex_planning', 'strategic_thinking']
                ),
                'llama3.1-uncensored': ModelCapability(
                    name='rolandroland/llama3.1-uncensored:latest',
                    size_gb=8.5,
                    domains=[TaskDomain.GENERAL, TaskDomain.REASONING, TaskDomain.CREATIVITY],
                    complexity_rating=8,
                    performance_history={},
                    usage_count=0,
                    success_rate=0.8,
                    average_response_time=2.8,
                    specialties=['unrestricted_thinking', 'creative_solutions', 'general_intelligence']
                ),
                'cogito': ModelCapability(
                    name='cogito:latest',
                    size_gb=8.5,
                    domains=[TaskDomain.REASONING, TaskDomain.ANALYSIS, TaskDomain.PLANNING],
                    complexity_rating=8,
                    performance_history={},
                    usage_count=0,
                    success_rate=0.82,
                    average_response_time=2.6,
                    specialties=['meta_cognitive_reasoning', 'self_analysis', 'philosophical_thinking']
                ),
                'codellama': ModelCapability(
                    name='codellama:7b',
                    size_gb=3.8,
                    domains=[TaskDomain.CODING, TaskDomain.DEBUGGING],
                    complexity_rating=7,
                    performance_history={},
                    usage_count=0,
                    success_rate=0.85,
                    average_response_time=2.0,
                    specialties=['code_generation', 'debugging', 'programming_best_practices']
                ),
                'llava': ModelCapability(
                    name='llava:7b',
                    size_gb=4.7,
                    domains=[TaskDomain.VISION, TaskDomain.ANALYSIS],
                    complexity_rating=6,
                    performance_history={},
                    usage_count=0,
                    success_rate=0.75,
                    average_response_time=3.5,
                    specialties=['image_analysis', 'visual_understanding', 'screenshot_interpretation']
                ),
                'mistral': ModelCapability(
                    name='mistral:7b',
                    size_gb=4.1,
                    domains=[TaskDomain.GENERAL, TaskDomain.REASONING],
                    complexity_rating=7,
                    performance_history={},
                    usage_count=0,
                    success_rate=0.78,
                    average_response_time=2.2,
                    specialties=['general_intelligence', 'quick_responses', 'versatile_tasks']
                ),
                'wizardlm-uncensored': ModelCapability(
                    name='wizardlm-uncensored:latest',
                    size_gb=7.4,
                    domains=[TaskDomain.CREATIVITY, TaskDomain.REASONING, TaskDomain.GENERAL],
                    complexity_rating=7,
                    performance_history={},
                    usage_count=0,
                    success_rate=0.77,
                    average_response_time=2.4,
                    specialties=['creative_problem_solving', 'unrestricted_analysis', 'innovative_thinking']
                ),
                'llama3.2': ModelCapability(
                    name='llama3.2:latest',
                    size_gb=2.0,
                    domains=[TaskDomain.GENERAL, TaskDomain.REASONING],
                    complexity_rating=6,
                    performance_history={},
                    usage_count=0,
                    success_rate=0.75,
                    average_response_time=1.8,
                    specialties=['fast_responses', 'lightweight_tasks', 'general_assistance']
                )
            }
            
            # Match available models to capabilities
            for model_data in raw_models.get('models', []):
                model_name = model_data.get('name', '')
                
                # Find matching capability definition
                capability = None
                for key, cap in model_mappings.items():
                    if key in model_name.lower() or cap.name == model_name:
                        capability = cap
                        capability.name = model_name  # Use exact name
                        break
                
                if capability:
                    self.models[model_name] = capability
                    logger.info(f"📋 Mapped {model_name} -> {capability.specialties}")
                else:
                    # Create generic capability for unknown models
                    self.models[model_name] = ModelCapability(
                        name=model_name,
                        size_gb=model_data.get('size', 0) / (1024**3),  # Convert to GB
                        domains=[TaskDomain.GENERAL],
                        complexity_rating=5,
                        performance_history={},
                        usage_count=0,
                        success_rate=0.5,
                        average_response_time=3.0,
                        specialties=['unknown_capabilities']
                    )
            
            logger.info(f"✅ Mapped {len(self.models)} models with capabilities")
            
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
    
    def get_best_model_for_task(self, task_analysis: TaskAnalysis) -> Optional[str]:
        """Get the best model for a specific task"""
        try:
            candidates = []
            
            for model_name, capability in self.models.items():
                score = 0
                
                # Domain match bonus
                if task_analysis.domain in capability.domains:
                    score += 50
                
                # Complexity match
                complexity_scores = {
                    TaskComplexity.SIMPLE: [1, 2, 3, 4, 5],
                    TaskComplexity.MODERATE: [4, 5, 6, 7],
                    TaskComplexity.COMPLEX: [7, 8, 9],
                    TaskComplexity.EXPERT: [9, 10]
                }
                
                if capability.complexity_rating in complexity_scores.get(task_analysis.complexity, []):
                    score += 30
                
                # Performance history
                score += capability.success_rate * 20
                
                # Penalize for slow response times if it's a simple task
                if task_analysis.complexity == TaskComplexity.SIMPLE:
                    score -= capability.average_response_time * 2
                
                candidates.append((model_name, score, capability))
            
            # Sort by score and return best
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            if candidates:
                best_model = candidates[0][0]
                logger.info(f"🎯 Selected {best_model} for {task_analysis.domain.value} task (score: {candidates[0][1]:.1f})")
                return best_model
            
            return None
            
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            return None
    
    def update_model_performance(self, model_name: str, success: bool, response_time: float):
        """Update model performance metrics"""
        if model_name in self.models:
            capability = self.models[model_name]
            capability.usage_count += 1
            
            # Update success rate with exponential moving average
            alpha = 0.1
            new_rate = 1.0 if success else 0.0
            capability.success_rate = (1 - alpha) * capability.success_rate + alpha * new_rate
            
            # Update response time
            capability.average_response_time = (
                (capability.average_response_time * (capability.usage_count - 1) + response_time) 
                / capability.usage_count
            )
            
            self.save_performance_data()
    
    def save_performance_data(self):
        """Save performance data to disk"""
        try:
            data = {}
            for name, capability in self.models.items():
                data[name] = {
                    'usage_count': capability.usage_count,
                    'success_rate': capability.success_rate,
                    'average_response_time': capability.average_response_time,
                    'performance_history': capability.performance_history
                }
            
            with open('model_performance.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save performance data: {e}")
    
    def load_performance_data(self):
        """Load performance data from disk"""
        try:
            if os.path.exists('model_performance.json'):
                with open('model_performance.json', 'r') as f:
                    data = json.load(f)
                    self.performance_cache = data
                    
        except Exception as e:
            logger.warning(f"Failed to load performance data: {e}")

class CapabilityGapAnalyzer:
    """Detects capability gaps and triggers self-improvement"""
    
    def __init__(self, agent):
        self.agent = agent
        self.failure_patterns = []
        self.improvement_history = []
        self.load_failure_patterns()
    
    def analyze_failure(self, task: str, error: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a failure to determine what capability is missing"""
        try:
            logger.info("🔍 Analyzing capability gap...")
            
            # Detect common failure patterns
            gap_type = self.classify_gap(task, error)
            
            # Record the failure
            failure_record = {
                'task': task,
                'error': error,
                'gap_type': gap_type,
                'timestamp': datetime.now().isoformat(),
                'context': context
            }
            
            self.failure_patterns.append(failure_record)
            self.save_failure_patterns()
            
            # Determine if we should trigger deepcoder consultation
            if self.should_consult_deepcoder(gap_type, task):
                logger.info("🚀 Triggering deepcoder consultation for capability development...")
                return self.trigger_capability_development(task, error, gap_type)
            
            return {
                'gap_detected': True,
                'gap_type': gap_type,
                'should_develop': False,
                'reason': 'Low priority or insufficient failure frequency'
            }
            
        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            return {'gap_detected': False, 'error': str(e)}
    
    def classify_gap(self, task: str, error: str) -> str:
        """Classify the type of capability gap"""
        task_lower = task.lower()
        error_lower = error.lower()
        
        # Web automation gaps
        if any(word in task_lower for word in ['web', 'browser', 'website', 'chatgpt', 'click', 'type']):
            if any(phrase in error_lower for phrase in ['selenium', 'webdriver', 'browser', 'element']):
                return 'web_automation'
            elif "can't" in error_lower or "unable" in error_lower:
                return 'web_interaction'
        
        # Coding gaps
        if any(word in task_lower for word in ['code', 'program', 'script']):
            if any(phrase in error_lower for phrase in ['import', 'module', 'library']):
                return 'missing_dependency'
            elif any(phrase in error_lower for phrase in ['syntax', 'error', 'exception']):
                return 'coding_skill'
        
        # API/Integration gaps
        if any(word in task_lower for word in ['api', 'connect', 'integrate']):
            return 'api_integration'
        
        # File/System gaps
        if any(word in task_lower for word in ['file', 'directory', 'system']):
            return 'file_system'
        
        # General capability gap
        if "can't" in error_lower or "unable" in error_lower:
            return 'general_capability'
        
        return 'unknown'
    
    def should_consult_deepcoder(self, gap_type: str, task: str) -> bool:
        """Determine if we should consult deepcoder for this gap"""
        # Always consult for web automation gaps (high value)
        if gap_type in ['web_automation', 'web_interaction']:
            return True
        
        # Consult for API integration gaps
        if gap_type == 'api_integration':
            return True
        
        # Consult for repeated failures of the same type
        similar_failures = [f for f in self.failure_patterns if f['gap_type'] == gap_type]
        if len(similar_failures) >= 2:
            return True
        
        # Consult for complex tasks
        if any(word in task.lower() for word in ['complex', 'advanced', 'sophisticated']):
            return True
        
        return False
    
    def trigger_capability_development(self, task: str, error: str, gap_type: str) -> Dict[str, Any]:
        """Trigger deepcoder to develop missing capability"""
        try:
            logger.info(f"🧠 Consulting deepcoder for {gap_type} capability...")
            
            # Get deepcoder interface
            deepcoder_interface = DeepCoderInterface(self.agent)
            
            # Request capability development
            development_result = deepcoder_interface.develop_capability(task, error, gap_type)
            
            if development_result.get('success', False):
                logger.info("✅ Successfully developed new capability!")
                
                # Record successful improvement
                improvement_record = {
                    'task': task,
                    'gap_type': gap_type,
                    'capability_name': development_result.get('capability_name'),
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
                
                self.improvement_history.append(improvement_record)
                self.save_improvement_history()
                
                return {
                    'gap_detected': True,
                    'gap_type': gap_type,
                    'should_develop': True,
                    'development_result': development_result,
                    'capability_developed': True
                }
            else:
                logger.warning("❌ Failed to develop new capability")
                return {
                    'gap_detected': True,
                    'gap_type': gap_type,
                    'should_develop': True,
                    'development_result': development_result,
                    'capability_developed': False
                }
                
        except Exception as e:
            logger.error(f"Capability development failed: {e}")
            return {'gap_detected': True, 'error': str(e)}
    
    def save_failure_patterns(self):
        """Save failure patterns to disk"""
        try:
            # Keep only recent patterns (last 100)
            self.failure_patterns = self.failure_patterns[-100:]
            
            with open('failure_patterns.json', 'w') as f:
                json.dump(self.failure_patterns, f, indent=2, cls=CustomJSONEncoder)
                
        except Exception as e:
            logger.warning(f"Failed to save failure patterns: {e}")
    
    def load_failure_patterns(self):
        """Load failure patterns from disk"""
        try:
            if os.path.exists('failure_patterns.json'):
                with open('failure_patterns.json', 'r') as f:
                    self.failure_patterns = json.load(f)
                    
        except Exception as e:
            logger.warning(f"Failed to load failure patterns: {e}")
    
    def save_improvement_history(self):
        """Save improvement history to disk"""
        try:
            with open('improvement_history.json', 'w') as f:
                json.dump(self.improvement_history, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save improvement history: {e}")

class DeepCoderInterface:
    """Interface for consulting deepcoder model for capability development"""
    
    def __init__(self, agent):
        self.agent = agent
        self.deepcoder_model = 'deepcoder:latest'
    
    def develop_capability(self, task: str, error: str, gap_type: str) -> Dict[str, Any]:
        """Ask deepcoder to develop a missing capability"""
        try:
            logger.info(f"🧠 Consulting deepcoder for {gap_type} development...")
            
            # Create comprehensive prompt for deepcoder
            prompt = self.create_development_prompt(task, error, gap_type)
            
            # Consult deepcoder
            start_time = time.time()
            response = self.agent.ollama.generate(
                model=self.deepcoder_model,
                prompt=prompt,
                options={'num_predict': 1500, 'temperature': 0.3}
            )
            response_time = time.time() - start_time
            
            deepcoder_response = response['response']
            
            # Parse deepcoder's response and extract capability
            capability_info = self.parse_capability_response(deepcoder_response, gap_type)
            
            if capability_info.get('code'):
                # Create the capability
                success = self.create_capability(capability_info, gap_type)
                
                # Update deepcoder performance
                self.agent.model_manager.update_model_performance(
                    self.deepcoder_model, success, response_time
                )
                
                return {
                    'success': success,
                    'capability_name': capability_info.get('name'),
                    'code': capability_info.get('code'),
                    'description': capability_info.get('description'),
                    'deepcoder_response': deepcoder_response
                }
            else:
                return {
                    'success': False,
                    'error': 'No valid capability code generated',
                    'deepcoder_response': deepcoder_response
                }
                
        except Exception as e:
            logger.error(f"Deepcoder consultation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_development_prompt(self, task: str, error: str, gap_type: str) -> str:
        """Create a comprehensive prompt for deepcoder to develop missing capability"""
        prompt = f"""I am an intelligent agent that encountered a capability gap. I need you to help me develop a new capability to handle this type of task.

TASK THAT FAILED: {task}
ERROR ENCOUNTERED: {error}
GAP TYPE: {gap_type}

Based on this failure, please create a Python module that implements the missing capability. The module should:

1. Be self-contained and importable
2. Include proper error handling
3. Have clear function/class interfaces
4. Include basic testing capability
5. Follow Python best practices

For {gap_type} gaps, focus on:
"""
        
        # Add specific guidance based on gap type
        gap_specific_guidance = {
            'web_automation': """
- Creating Selenium WebDriver automation
- Element finding and interaction methods
- Page navigation and waiting strategies
- Screenshot capture and analysis capabilities
- Error handling for common web issues
""",
            'web_interaction': """
- Browser automation for complex interactions
- Form filling and submission
- JavaScript execution capabilities
- Cookie and session management
- Dynamic content handling
""",
            'api_integration': """
- HTTP client implementations
- Authentication handling
- Response parsing and error handling
- Rate limiting and retry logic
- Data transformation utilities
""",
            'file_system': """
- File and directory operations
- Path handling and validation
- Permission and access checking
- Backup and versioning capabilities
- Cross-platform compatibility
""",
            'coding_skill': """
- Code generation and analysis
- Syntax validation and correction
- Best practice enforcement
- Documentation generation
- Testing framework integration
""",
            'general_capability': """
- Modular design for extensibility
- Clear API interfaces
- Comprehensive error handling
- Logging and debugging support
- Performance optimization
"""
        }
        
        prompt += gap_specific_guidance.get(gap_type, gap_specific_guidance['general_capability'])
        
        prompt += f"""
Please provide:

1. **CAPABILITY_NAME**: A descriptive name for this capability
2. **DESCRIPTION**: Brief description of what this capability does
3. **CODE**: Complete Python code implementation

Format your response as:
```
CAPABILITY_NAME: [name]
DESCRIPTION: [description]
CODE:
```python
[your complete Python code here]
```
```

The code should be production-ready and handle the specific failure case mentioned above.
"""
        
        return prompt
    
    def parse_capability_response(self, response: str, gap_type: str) -> Dict[str, Any]:
        """Parse deepcoder's response to extract capability information"""
        try:
            capability_info = {
                'name': f'{gap_type}_capability_{int(time.time())}',
                'description': f'Auto-generated capability for {gap_type}',
                'code': ''
            }
            
            # Extract capability name
            name_match = re.search(r'CAPABILITY_NAME:\s*(.+)', response, re.IGNORECASE)
            if name_match:
                capability_info['name'] = name_match.group(1).strip()
            
            # Extract description
            desc_match = re.search(r'DESCRIPTION:\s*(.+)', response, re.IGNORECASE)
            if desc_match:
                capability_info['description'] = desc_match.group(1).strip()
            
            # Extract code
            code_patterns = [
                r'```python\n(.*?)\n```',
                r'CODE:\s*```python\n(.*?)\n```',
                r'CODE:\s*```\n(.*?)\n```',
                r'```\n(.*?)\n```'
            ]
            
            for pattern in code_patterns:
                code_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if code_match:
                    capability_info['code'] = code_match.group(1).strip()
                    break
            
            # If no code blocks found, try to extract any Python-looking code
            if not capability_info['code']:
                lines = response.split('\n')
                in_code = False
                code_lines = []
                
                for line in lines:
                    if any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from ']):
                        in_code = True
                    
                    if in_code:
                        code_lines.append(line)
                
                if code_lines:
                    capability_info['code'] = '\n'.join(code_lines)
            
            return capability_info
            
        except Exception as e:
            logger.error(f"Failed to parse capability response: {e}")
            return {'name': '', 'description': '', 'code': ''}
    
    def create_capability(self, capability_info: Dict[str, Any], gap_type: str) -> bool:
        """Create and integrate the new capability"""
        try:
            if not capability_info.get('code'):
                logger.error("No code provided for capability creation")
                return False
            
            capability_name = capability_info['name'].replace(' ', '_').lower()
            
            # Use the agent's capability manager to create the capability
            success = self.agent.capability_manager.create_capability(
                capability_name,
                capability_info['description'],
                capability_info['code']
            )
            
            if success:
                logger.info(f"✅ Successfully created and integrated capability: {capability_name}")
                
                # Add to agent's self-model
                if hasattr(self.agent, 'meta_cognition'):
                    self.agent.meta_cognition.self_model['learned_skills'].append(capability_name)
                    self.agent.meta_cognition.self_model['current_capabilities'].append({
                        'name': capability_name,
                        'description': capability_info['description'],
                        'gap_type': gap_type,
                        'created': datetime.now().isoformat()
                    })
                    self.agent.meta_cognition.save_self_model()
                
                return True
            else:
                logger.error(f"Failed to create capability: {capability_name}")
                return False
                
        except Exception as e:
            logger.error(f"Capability creation failed: {e}")
            return False

class WebMasterPro:
    """Advanced web automation and interaction capabilities"""
    
    def __init__(self, agent):
        self.agent = agent
        self.driver = None
        self.wait = None
    
    def initialize_browser(self, headless: bool = False) -> bool:
        """Initialize web browser for automation"""
        try:
            if not SELENIUM_AVAILABLE:
                logger.error("Selenium not available for web automation")
                return False
            
            options = Options()
            if headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            
            self.driver = webdriver.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, 10)
            
            logger.info("✅ Browser initialized for web automation")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            return False
    
    def navigate_and_interact(self, url: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Navigate to URL and perform a series of interactions"""
        try:
            if not self.driver:
                if not self.initialize_browser():
                    return {'success': False, 'error': 'Failed to initialize browser'}
            
            # Navigate to URL
            logger.info(f"🌐 Navigating to: {url}")
            self.driver.get(url)
            time.sleep(2)  # Wait for page load
            
            results = []
            
            for i, action in enumerate(actions):
                logger.info(f"🎯 Executing action {i+1}: {action.get('type', 'unknown')}")
                
                action_result = self.execute_action(action)
                results.append(action_result)
                
                if not action_result.get('success', False):
                    logger.warning(f"Action {i+1} failed: {action_result.get('error', 'Unknown error')}")
                
                time.sleep(1)  # Small delay between actions
            
            # Take screenshot for analysis
            screenshot_path = f"screenshots/interaction_{int(time.time())}.png"
            os.makedirs("screenshots", exist_ok=True)
            self.driver.save_screenshot(screenshot_path)
            
            return {
                'success': True,
                'results': results,
                'screenshot': screenshot_path,
                'page_title': self.driver.title,
                'current_url': self.driver.current_url
            }
            
        except Exception as e:
            logger.error(f"Web interaction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single web action"""
        try:
            action_type = action.get('type', '')
            
            if action_type == 'click':
                return self.click_element(action)
            elif action_type == 'type':
                return self.type_text(action)
            elif action_type == 'wait':
                return self.wait_for_element(action)
            elif action_type == 'scroll':
                return self.scroll_page(action)
            elif action_type == 'screenshot':
                return self.take_screenshot(action)
            else:
                return {'success': False, 'error': f'Unknown action type: {action_type}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def click_element(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Click on an element"""
        try:
            selector = action.get('selector', '')
            selector_type = action.get('selector_type', 'css')
            
            if selector_type == 'css':
                element = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            elif selector_type == 'xpath':
                element = self.wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
            elif selector_type == 'id':
                element = self.wait.until(EC.element_to_be_clickable((By.ID, selector)))
            else:
                return {'success': False, 'error': f'Unsupported selector type: {selector_type}'}
            
            element.click()
            return {'success': True, 'action': 'click', 'selector': selector}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def type_text(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Type text into an element"""
        try:
            selector = action.get('selector', '')
            text = action.get('text', '')
            selector_type = action.get('selector_type', 'css')
            
            if selector_type == 'css':
                element = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            elif selector_type == 'xpath':
                element = self.wait.until(EC.presence_of_element_located((By.XPATH, selector)))
            elif selector_type == 'id':
                element = self.wait.until(EC.presence_of_element_located((By.ID, selector)))
            else:
                return {'success': False, 'error': f'Unsupported selector type: {selector_type}'}
            
            element.clear()
            element.send_keys(text)
            
            # Press Enter if specified
            if action.get('press_enter', False):
                element.send_keys(Keys.RETURN)
            
            return {'success': True, 'action': 'type', 'text': text, 'selector': selector}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def wait_for_element(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for an element to appear"""
        try:
            selector = action.get('selector', '')
            timeout = action.get('timeout', 10)
            selector_type = action.get('selector_type', 'css')
            
            wait = WebDriverWait(self.driver, timeout)
            
            if selector_type == 'css':
                element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            elif selector_type == 'xpath':
                element = wait.until(EC.presence_of_element_located((By.XPATH, selector)))
            elif selector_type == 'id':
                element = wait.until(EC.presence_of_element_located((By.ID, selector)))
            else:
                return {'success': False, 'error': f'Unsupported selector type: {selector_type}'}
            
            return {'success': True, 'action': 'wait', 'selector': selector, 'found': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def scroll_page(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Scroll the page"""
        try:
            direction = action.get('direction', 'down')
            amount = action.get('amount', 500)
            
            if direction == 'down':
                self.driver.execute_script(f"window.scrollBy(0, {amount});")
            elif direction == 'up':
                self.driver.execute_script(f"window.scrollBy(0, -{amount});")
            elif direction == 'top':
                self.driver.execute_script("window.scrollTo(0, 0);")
            elif direction == 'bottom':
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            return {'success': True, 'action': 'scroll', 'direction': direction}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def take_screenshot(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Take a screenshot"""
        try:
            filename = action.get('filename', f'screenshot_{int(time.time())}.png')
            directory = action.get('directory', 'screenshots')
            
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
            
            self.driver.save_screenshot(filepath)
            
            return {'success': True, 'action': 'screenshot', 'filepath': filepath}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def close_browser(self):
        """Close the browser"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                self.wait = None
                logger.info("🔒 Browser closed")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")

class TaskAnalyzer:
    """Analyzes tasks to determine complexity, domain, and execution strategy"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def analyze_task(self, task_description: str) -> TaskAnalysis:
        """Analyze a task and return structured analysis"""
        try:
            # Determine domain
            domain = self.classify_domain(task_description)
            
            # Determine complexity
            complexity = self.assess_complexity(task_description)
            
            # Estimate steps
            estimated_steps = self.estimate_steps(task_description, complexity)
            
            # Identify required capabilities
            required_capabilities = self.identify_capabilities(task_description, domain)
            
            # Assess confidence
            confidence = self.assess_confidence(task_description, domain, required_capabilities)
            
            # Generate alternative approaches
            alternatives = self.generate_alternatives(task_description, domain)
            
            return TaskAnalysis(
                description=task_description,
                complexity=complexity,
                domain=domain,
                estimated_steps=estimated_steps,
                required_capabilities=required_capabilities,
                confidence=confidence,
                alternative_approaches=alternatives
            )
            
        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
            return TaskAnalysis(
                description=task_description,
                complexity=TaskComplexity.SIMPLE,
                domain=TaskDomain.GENERAL,
                estimated_steps=1,
                required_capabilities=[],
                confidence=0.5,
                alternative_approaches=[]
            )
    
    def classify_domain(self, task: str) -> TaskDomain:
        """Classify the domain of a task"""
        task_lower = task.lower()
        
        domain_keywords = {
            TaskDomain.WEB_AUTOMATION: ['web', 'browser', 'website', 'url', 'click', 'type', 'navigate', 'chatgpt', 'google'],
            TaskDomain.CODING: ['code', 'program', 'script', 'function', 'class', 'python', 'javascript'],
            TaskDomain.REASONING: ['analyze', 'think', 'reason', 'logic', 'problem', 'solve'],
            TaskDomain.PLANNING: ['plan', 'strategy', 'organize', 'schedule', 'steps', 'workflow'],
            TaskDomain.ANALYSIS: ['data', 'statistics', 'research', 'investigate', 'examine'],
            TaskDomain.CREATIVITY: ['create', 'design', 'generate', 'invent', 'brainstorm', 'art'],
            TaskDomain.DEBUGGING: ['debug', 'fix', 'error', 'bug', 'troubleshoot', 'repair'],
            TaskDomain.SYSTEM: ['file', 'directory', 'install', 'system', 'command', 'terminal'],
            TaskDomain.VISION: ['image', 'picture', 'visual', 'screenshot', 'analyze'],
        }
        
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if score > 0:
                scores[domain] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return TaskDomain.GENERAL
    
    def assess_complexity(self, task: str) -> TaskComplexity:
        """Assess the complexity of a task"""
        task_lower = task.lower()
        
        complexity_indicators = {
            TaskComplexity.EXPERT: ['advanced', 'complex', 'sophisticated', 'enterprise', 'scalable'],
            TaskComplexity.COMPLEX: ['multiple', 'integrate', 'coordinate', 'orchestrate', 'comprehensive'],
            TaskComplexity.MODERATE: ['create', 'build', 'develop', 'implement', 'configure'],
            TaskComplexity.SIMPLE: ['open', 'click', 'type', 'read', 'simple', 'basic']
        }
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in task_lower for indicator in indicators):
                return complexity
        
        # Count verbs as complexity indicator
        verb_count = len([word for word in task_lower.split() if word in ['and', 'then', 'after', 'before', 'next']])
        
        if verb_count >= 3:
            return TaskComplexity.COMPLEX
        elif verb_count >= 2:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def estimate_steps(self, task: str, complexity: TaskComplexity) -> int:
        """Estimate the number of steps required"""
        base_steps = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 3,
            TaskComplexity.COMPLEX: 7,
            TaskComplexity.EXPERT: 12
        }
        
        # Count conjunctions and sequence words
        sequence_words = ['and', 'then', 'after', 'next', 'also', 'additionally']
        sequence_count = sum(1 for word in sequence_words if word in task.lower())
        
        return base_steps[complexity] + sequence_count
    
    def identify_capabilities(self, task: str, domain: TaskDomain) -> List[str]:
        """Identify required capabilities for the task"""
        task_lower = task.lower()
        capabilities = []
        
        capability_map = {
            'web_automation': ['web', 'browser', 'website', 'click', 'type'],
            'file_operations': ['file', 'create', 'write', 'save', 'read'],
            'network_requests': ['api', 'http', 'request', 'download', 'fetch'],
            'data_processing': ['process', 'analyze', 'parse', 'transform'],
            'system_automation': ['system', 'command', 'install', 'run'],
            'code_generation': ['code', 'program', 'script', 'function'],
            'image_processing': ['image', 'picture', 'screenshot', 'visual']
        }
        
        for capability, keywords in capability_map.items():
            if any(keyword in task_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities
    
    def assess_confidence(self, task: str, domain: TaskDomain, capabilities: List[str]) -> float:
        """Assess confidence in completing the task"""
        # Base confidence from agent's self-model
        base_confidence = self.agent.meta_cognition.assess_capability_for_task(task)
        
        # Adjust based on domain familiarity
        domain_confidence = {
            TaskDomain.GENERAL: 0.8,
            TaskDomain.REASONING: 0.7,
            TaskDomain.CODING: 0.6,
            TaskDomain.WEB_AUTOMATION: 0.4,  # May need new capabilities
            TaskDomain.SYSTEM: 0.6,
            TaskDomain.VISION: 0.3  # May need special tools
        }
        
        domain_modifier = domain_confidence.get(domain, 0.5)
        
        # Adjust based on capability requirements
        capability_penalty = len(capabilities) * 0.1
        
        final_confidence = min(1.0, base_confidence * domain_modifier - capability_penalty)
        return max(0.1, final_confidence)
    
    def generate_alternatives(self, task: str, domain: TaskDomain) -> List[str]:
        """Generate alternative approaches for the task"""
        alternatives = []
        
        if domain == TaskDomain.WEB_AUTOMATION:
            alternatives.extend([
                "Use web scraping with requests/BeautifulSoup",
                "Implement browser automation with Selenium",
                "Try API integration if available"
            ])
        elif domain == TaskDomain.CODING:
            alternatives.extend([
                "Generate code step by step",
                "Use existing code templates",
                "Break into smaller functions"
            ])
        
        return alternatives

class AIHelper:
    """Individual AI model helper for specialized tasks"""
    
    def __init__(self, model_name: str, specialty: str, ollama_client):
        self.model_name = model_name
        self.specialty = specialty
        self.ollama = ollama_client
        self.conversation_history = []
        self.performance_score = 0.5
        self.usage_count = 0
    
    def think(self, prompt: str, context: str = "") -> str:
        """Have this AI helper think about a problem"""
        try:
            full_prompt = f"""You are a specialized AI assistant with expertise in: {self.specialty}

Context: {context}

Task: {prompt}

Think deeply about this and provide your best analysis, solution, or recommendation. Be specific and actionable.
"""
            
            response = self.ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={'num_predict': 800, 'temperature': 0.7}
            )
            
            result = response['response']
            self.conversation_history.append({'prompt': prompt, 'response': result})
            self.usage_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"AI Helper {self.model_name} failed: {e}")
            return f"Error: {e}"
    
    def update_performance(self, success: bool):
        """Update performance score based on success/failure"""
        if self.usage_count == 0:
            self.performance_score = 1.0 if success else 0.0
        else:
            # Exponential moving average
            alpha = 0.1
            new_score = 1.0 if success else 0.0
            self.performance_score = (1 - alpha) * self.performance_score + alpha * new_score

class TaskGoalValidator:
    """Validates whether the actual goal of a task has been achieved"""
    
    def __init__(self, agent):
        self.agent = agent
        self.goal_patterns = {}
        self.load_goal_patterns()
    
    def define_success_criteria(self, task_description: str) -> GoalCriteria:
        """Define what success actually means for this specific task"""
        try:
            task_lower = task_description.lower()
            
            # Web automation goals
            if any(word in task_lower for word in ['chatgpt', 'openai']) and 'message' in task_lower:
                return GoalCriteria(
                    task_description=task_description,
                    success_indicators=['message_sent_successfully', 'response_received', 'response_content_read'],
                    evidence_required=['message_in_chat', 'assistant_response_present', 'response_text_extracted'],
                    validation_methods=['check_message_sent', 'verify_response_received', 'extract_response_text'],
                    failure_indicators=['element_not_interactable', 'timeout', 'no_response_detected']
                )
            
            # General web automation
            elif any(word in task_lower for word in ['website', 'web', 'browser', 'click', 'type']):
                return GoalCriteria(
                    task_description=task_description,
                    success_indicators=['page_loaded', 'interaction_completed', 'expected_result_achieved'],
                    evidence_required=['screenshot_taken', 'page_title_captured', 'interaction_confirmed'],
                    validation_methods=['verify_page_load', 'check_interaction_success', 'validate_result'],
                    failure_indicators=['page_not_found', 'element_not_found', 'interaction_failed']
                )
            
            # Coding tasks
            elif any(word in task_lower for word in ['code', 'program', 'script', 'function']):
                return GoalCriteria(
                    task_description=task_description,
                    success_indicators=['code_created', 'code_runs_successfully', 'requirements_met'],
                    evidence_required=['file_created', 'execution_successful', 'output_correct'],
                    validation_methods=['check_file_exists', 'test_execution', 'verify_functionality'],
                    failure_indicators=['syntax_error', 'runtime_error', 'incorrect_output']
                )
            
            # File operations
            elif any(word in task_lower for word in ['file', 'create', 'write', 'save']):
                return GoalCriteria(
                    task_description=task_description,
                    success_indicators=['file_created', 'content_written', 'file_accessible'],
                    evidence_required=['file_exists', 'content_correct', 'permissions_valid'],
                    validation_methods=['check_file_exists', 'verify_content', 'test_access'],
                    failure_indicators=['creation_failed', 'write_error', 'permission_denied']
                )
            
            # System automation
            elif any(word in task_lower for word in ['open', 'launch', 'run', 'execute']):
                return GoalCriteria(
                    task_description=task_description,
                    success_indicators=['application_launched', 'command_executed', 'action_completed'],
                    evidence_required=['process_running', 'output_captured', 'result_confirmed'],
                    validation_methods=['check_process', 'verify_output', 'confirm_result'],
                    failure_indicators=['launch_failed', 'command_error', 'execution_timeout']
                )
            
            # General tasks
            else:
                return GoalCriteria(
                    task_description=task_description,
                    success_indicators=['task_completed', 'objective_achieved'],
                    evidence_required=['result_produced', 'output_generated'],
                    validation_methods=['verify_completion', 'check_objective'],
                    failure_indicators=['task_failed', 'no_progress']
                )
                
        except Exception as e:
            logger.error(f"Failed to define success criteria: {e}")
            return GoalCriteria(
                task_description=task_description,
                success_indicators=['basic_completion'],
                evidence_required=['some_result'],
                validation_methods=['basic_check'],
                failure_indicators=['complete_failure']
            )
    
    def validate_goal_achievement(self, task_description: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate whether the goal has actually been achieved"""
        try:
            # Get success criteria for this task
            criteria = self.define_success_criteria(task_description)
            
            validation_result = {
                'goal_achieved': False,
                'criteria_met': [],
                'criteria_failed': [],
                'evidence_found': [],
                'evidence_missing': [],
                'confidence_score': 0.0,
                'next_actions': []
            }
            
            # Check each success indicator
            for indicator in criteria.success_indicators:
                is_met = self.check_success_indicator(indicator, execution_result, task_description)
                if is_met:
                    validation_result['criteria_met'].append(indicator)
                else:
                    validation_result['criteria_failed'].append(indicator)
            
            # Check for required evidence
            for evidence in criteria.evidence_required:
                is_found = self.check_evidence(evidence, execution_result)
                if is_found:
                    validation_result['evidence_found'].append(evidence)
                else:
                    validation_result['evidence_missing'].append(evidence)
            
            # Check for failure indicators
            failure_detected = False
            for failure_indicator in criteria.failure_indicators:
                if self.check_failure_indicator(failure_indicator, execution_result):
                    failure_detected = True
                    validation_result['criteria_failed'].append(f"failure_detected: {failure_indicator}")
            
            # Calculate confidence score
            total_criteria = len(criteria.success_indicators)
            met_criteria = len(validation_result['criteria_met'])
            total_evidence = len(criteria.evidence_required)
            found_evidence = len(validation_result['evidence_found'])
            
            if total_criteria > 0:
                criteria_score = met_criteria / total_criteria
            else:
                criteria_score = 0.0
                
            if total_evidence > 0:
                evidence_score = found_evidence / total_evidence
            else:
                evidence_score = 1.0  # No evidence required
            
            # Penalize for failure indicators
            failure_penalty = 0.5 if failure_detected else 0.0
            
            validation_result['confidence_score'] = max(0.0, (criteria_score * 0.6 + evidence_score * 0.4) - failure_penalty)
            
            # Determine if goal is achieved
            validation_result['goal_achieved'] = (
                validation_result['confidence_score'] >= 0.7 and
                not failure_detected and
                len(validation_result['criteria_failed']) == 0
            )
            
            # Generate next actions if goal not achieved
            if not validation_result['goal_achieved']:
                validation_result['next_actions'] = self.generate_next_actions(
                    task_description, criteria, validation_result
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Goal validation failed: {e}")
            return {
                'goal_achieved': False,
                'error': str(e),
                'confidence_score': 0.0,
                'next_actions': ['retry_with_different_approach']
            }
    
    def check_success_indicator(self, indicator: str, execution_result: Dict[str, Any], task_description: str) -> bool:
        """Check if a specific success indicator is met"""
        try:
            result_str = str(execution_result).lower()
            task_lower = task_description.lower()
            
            # Web automation indicators
            if indicator == 'message_sent_successfully':
                return (
                    execution_result.get('success', False) and
                    'sent' in result_str and
                    'message' in result_str and
                    'error' not in result_str
                )
            
            elif indicator == 'response_received':
                return (
                    'response' in result_str and
                    ('received' in result_str or 'replied' in result_str or 'answered' in result_str)
                )
            
            elif indicator == 'response_content_read':
                return (
                    'content' in result_str or
                    'text' in result_str or
                    'said' in result_str or
                    len(str(execution_result.get('response', ''))) > 10
                )
            
            # Coding indicators
            elif indicator == 'code_created':
                return (
                    execution_result.get('success', False) and
                    ('file' in result_str or 'code' in result_str) and
                    'created' in result_str
                )
            
            elif indicator == 'code_runs_successfully':
                return (
                    execution_result.get('success', False) and
                    'error' not in result_str and
                    ('executed' in result_str or 'ran' in result_str or 'success' in result_str)
                )
            
            # File operation indicators
            elif indicator == 'file_created':
                return (
                    execution_result.get('success', False) and
                    ('file' in result_str and 'created' in result_str)
                )
            
            # General indicators
            elif indicator == 'task_completed':
                return execution_result.get('success', False)
            
            elif indicator == 'page_loaded':
                return (
                    execution_result.get('success', False) and
                    ('page' in result_str or 'loaded' in result_str or 'title' in result_str)
                )
            
            elif indicator == 'interaction_completed':
                return (
                    execution_result.get('success', False) and
                    ('click' in result_str or 'type' in result_str or 'interaction' in result_str)
                )
            
            # Default: check for success flag
            return execution_result.get('success', False)
            
        except Exception as e:
            logger.error(f"Failed to check success indicator {indicator}: {e}")
            return False
    
    def check_evidence(self, evidence: str, execution_result: Dict[str, Any]) -> bool:
        """Check if required evidence is present"""
        try:
            # Screenshot evidence
            if evidence == 'screenshot_taken':
                return (
                    'screenshot' in execution_result or
                    'image' in str(execution_result) or
                    '.png' in str(execution_result)
                )
            
            # File evidence
            elif evidence == 'file_created':
                return (
                    'filename' in execution_result or
                    'file' in str(execution_result) or
                    os.path.exists(execution_result.get('filename', ''))
                )
            
            # Response evidence
            elif evidence == 'response_text_extracted':
                response_text = execution_result.get('response', '')
                return len(str(response_text)) > 5  # Has meaningful response
            
            # Execution evidence
            elif evidence == 'execution_successful':
                return execution_result.get('success', False) and 'error' not in str(execution_result)
            
            # General result evidence
            elif evidence == 'result_produced':
                return len(str(execution_result)) > 50  # Has substantial result
            
            return True  # Default to true for unknown evidence types
            
        except Exception as e:
            logger.error(f"Failed to check evidence {evidence}: {e}")
            return False
    
    def check_failure_indicator(self, indicator: str, execution_result: Dict[str, Any]) -> bool:
        """Check if a failure indicator is present"""
        try:
            result_str = str(execution_result).lower()
            
            failure_patterns = {
                'element_not_interactable': ['element not interactable', 'not clickable', 'not found'],
                'timeout': ['timeout', 'timed out', 'time limit exceeded'],
                'no_response_detected': ['no response', 'empty response', 'failed to read'],
                'syntax_error': ['syntax error', 'invalid syntax'],
                'runtime_error': ['runtime error', 'exception', 'traceback'],
                'permission_denied': ['permission denied', 'access denied', 'forbidden'],
                'page_not_found': ['404', 'not found', 'page does not exist'],
                'connection_error': ['connection', 'network', 'unreachable']
            }
            
            patterns = failure_patterns.get(indicator, [indicator])
            return any(pattern in result_str for pattern in patterns)
            
        except Exception as e:
            logger.error(f"Failed to check failure indicator {indicator}: {e}")
            return False
    
    def generate_next_actions(self, task_description: str, criteria: GoalCriteria, validation_result: Dict[str, Any]) -> List[str]:
        """Generate next actions to achieve the goal"""
        try:
            next_actions = []
            
            # If criteria failed, suggest specific approaches
            for failed_criterion in validation_result['criteria_failed']:
                if 'message_sent' in failed_criterion:
                    next_actions.extend([
                        'try_different_selector_for_input',
                        'wait_longer_for_element',
                        'use_javascript_to_interact',
                        'try_alternative_chatgpt_interface'
                    ])
                elif 'response_received' in failed_criterion:
                    next_actions.extend([
                        'wait_longer_for_response', 
                        'check_different_response_selectors',
                        'scroll_to_find_response',
                        'use_api_instead_of_web'
                    ])
                elif 'code_runs' in failed_criterion:
                    next_actions.extend([
                        'debug_and_fix_code',
                        'try_simpler_implementation',
                        'install_missing_dependencies',
                        'use_different_approach'
                    ])
            
            # If evidence is missing, suggest evidence collection
            for missing_evidence in validation_result['evidence_missing']:
                if 'screenshot' in missing_evidence:
                    next_actions.append('take_screenshot_for_verification')
                elif 'response_text' in missing_evidence:
                    next_actions.append('extract_response_text_with_different_method')
                elif 'file' in missing_evidence:
                    next_actions.append('verify_file_creation_and_content')
            
            # Generic fallback actions
            if not next_actions:
                next_actions = [
                    'try_alternative_approach',
                    'consult_deepcoder_for_solution',
                    'break_task_into_smaller_steps',
                    'use_different_tools_or_methods'
                ]
            
            return next_actions[:3]  # Return top 3 actions
            
        except Exception as e:
            logger.error(f"Failed to generate next actions: {e}")
            return ['retry_with_different_approach']
    
    def save_goal_patterns(self):
        """Save learned goal patterns"""
        try:
            with open('goal_patterns.json', 'w') as f:
                json.dump(self.goal_patterns, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save goal patterns: {e}")
    
    def load_goal_patterns(self):
        """Load learned goal patterns"""
        try:
            if os.path.exists('goal_patterns.json'):
                with open('goal_patterns.json', 'r') as f:
                    self.goal_patterns = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load goal patterns: {e}")

class PersistentDebugger:
    """Implements persistent debugging with multiple strategies until goal is achieved"""
    
    def __init__(self, agent):
        self.agent = agent
        self.strategy_history = []
        self.max_attempts = 5
        self.load_strategy_history()
    
    def execute_with_persistence(self, task_description: str, task_analysis: TaskAnalysis) -> Dict[str, Any]:
        """Execute task with persistent debugging until goal is achieved"""
        try:
            logger.info(f"🎯 Executing with persistence: {task_description[:50]}...")
            
            # Get goal validator
            goal_validator = self.agent.goal_validator
            
            # Get strategy engine for this task domain
            strategy_engine = self.agent.strategy_engine
            strategies = strategy_engine.get_strategies_for_domain(task_analysis.domain)
            
            attempts = []
            goal_achieved = False
            
            for attempt_num in range(self.max_attempts):
                logger.info(f"🔄 Attempt {attempt_num + 1}/{self.max_attempts}")
                
                # Get next strategy
                if attempt_num < len(strategies):
                    strategy = strategies[attempt_num]
                else:
                    # Generate new strategy using AI
                    strategy = self.generate_emergency_strategy(task_description, attempts)
                
                logger.info(f"🛠️ Using strategy: {strategy['name']}")
                
                # Execute strategy
                start_time = time.time()
                execution_result = self.execute_strategy(strategy, task_description, task_analysis)
                execution_time = time.time() - start_time
                
                # Validate goal achievement
                validation_result = goal_validator.validate_goal_achievement(task_description, execution_result)
                
                # Record attempt
                attempt = StrategyAttempt(
                    strategy_name=strategy['name'],
                    approach=strategy['description'],
                    parameters=strategy.get('parameters', {}),
                    result=StrategyResult.SUCCESS if validation_result['goal_achieved'] else StrategyResult.FAILURE,
                    execution_details=execution_result,
                    evidence_collected=validation_result.get('evidence_found', []),
                    failure_reason='; '.join(validation_result.get('criteria_failed', [])),
                    execution_time=execution_time
                )
                
                attempts.append(attempt)
                
                # Check if goal achieved
                if validation_result['goal_achieved']:
                    logger.info(f"✅ Goal achieved on attempt {attempt_num + 1}!")
                    goal_achieved = True
                    
                    # Learn from successful strategy
                    self.learn_from_success(strategy, task_analysis.domain, execution_result)
                    break
                else:
                    logger.info(f"❌ Attempt {attempt_num + 1} failed. Goal not achieved.")
                    logger.info(f"Failed criteria: {validation_result.get('criteria_failed', [])}")
                    logger.info(f"Next actions suggested: {validation_result.get('next_actions', [])}")
                    
                    # If this was the last attempt, try deepcoder
                    if attempt_num == self.max_attempts - 1:
                        logger.info("🧠 Final attempt failed, consulting deepcoder...")
                        deepcoder_result = self.consult_deepcoder_for_solution(task_description, attempts)
                        
                        if deepcoder_result.get('success', False):
                            logger.info("🚀 Deepcoder provided solution, retrying...")
                            final_result = self.execute_strategy(
                                deepcoder_result['strategy'], 
                                task_description, 
                                task_analysis
                            )
                            
                            final_validation = goal_validator.validate_goal_achievement(task_description, final_result)
                            if final_validation['goal_achieved']:
                                goal_achieved = True
                                execution_result = final_result
            
            # Compile final result
            final_result = {
                'goal_achieved': goal_achieved,
                'total_attempts': len(attempts),
                'successful_strategy': attempts[-1].strategy_name if goal_achieved else None,
                'execution_result': execution_result if goal_achieved else attempts[-1].execution_details,
                'attempt_history': [asdict(attempt) for attempt in attempts],
                'persistence_summary': self.generate_persistence_summary(attempts, goal_achieved)
            }
            
            # Save strategy history for learning
            self.save_strategy_history()
            
            return final_result
            
        except Exception as e:
            logger.error(f"Persistent execution failed: {e}")
            return {
                'goal_achieved': False,
                'error': str(e),
                'total_attempts': 0,
                'persistence_summary': 'Execution failed due to system error'
            }
    
    def execute_strategy(self, strategy: Dict[str, Any], task_description: str, task_analysis: TaskAnalysis) -> Dict[str, Any]:
        """Execute a specific strategy"""
        try:
            strategy_type = strategy.get('type', 'general')
            
            if strategy_type == 'web_automation':
                return self.execute_web_strategy(strategy, task_description)
            elif strategy_type == 'coding':
                return self.execute_coding_strategy(strategy, task_description)
            elif strategy_type == 'file_operation':
                return self.execute_file_strategy(strategy, task_description)
            elif strategy_type == 'system_automation':
                return self.execute_system_strategy(strategy, task_description)
            elif strategy_type == 'deepcoder_generated':
                return self.execute_deepcoder_strategy(strategy, task_description)
            else:
                # Use original execution method as fallback
                return self.agent.execute_with_context(
                    task_description, task_analysis, 
                    strategy.get('context', ''), 
                    strategy.get('model', None)
                )
                
        except Exception as e:
            return {'success': False, 'error': f"Strategy execution failed: {e}"}
    
    def execute_web_strategy(self, strategy: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Execute web automation strategy with specific parameters"""
        try:
            parameters = strategy.get('parameters', {})
            
            # Enhanced web automation with strategy-specific parameters
            if 'chatgpt' in task_description.lower():
                # ChatGPT-specific strategies
                if strategy['name'] == 'advanced_chatgpt_interaction':
                    actions = [
                        {'type': 'wait', 'selector': parameters.get('input_selector', 'textarea'), 'timeout': 15},
                        {'type': 'click', 'selector': parameters.get('input_selector', 'textarea')},
                        {'type': 'type', 'selector': parameters.get('input_selector', 'textarea'), 
                         'text': parameters.get('message', 'Hey'), 'press_enter': True},
                        {'type': 'wait', 'selector': parameters.get('response_selector', '[data-message-author-role="assistant"]'), 'timeout': 45}
                    ]
                elif strategy['name'] == 'javascript_chatgpt_interaction':
                    actions = [
                        {'type': 'execute_js', 'script': parameters.get('js_script', self.get_chatgpt_js_script())},
                        {'type': 'wait', 'timeout': 30}
                    ]
                else:
                    # Default ChatGPT actions
                    actions = [
                        {'type': 'wait', 'selector': 'textarea', 'timeout': 10},
                        {'type': 'type', 'selector': 'textarea', 'text': 'Hey', 'press_enter': True},
                        {'type': 'wait', 'selector': '[data-message-author-role="assistant"]', 'timeout': 30}
                    ]
                
                return self.agent.web_master.navigate_and_interact('https://chatgpt.com', actions)
            
            else:
                # Generic web strategy
                return self.agent.handle_web_automation_task(task_description, strategy.get('context', ''))
                
        except Exception as e:
            return {'success': False, 'error': f"Web strategy execution failed: {e}"}
    
    def execute_coding_strategy(self, strategy: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Execute coding strategy with specific parameters"""
        try:
            parameters = strategy.get('parameters', {})
            
            # Use specified model or best coding helper
            model = parameters.get('model')
            if model:
                coding_helper = None
                for helper in self.agent.ai_helpers.values():
                    if helper.model_name == model:
                        coding_helper = helper
                        break
                if not coding_helper:
                    coding_helper = self.agent.get_best_helper('coding')
            else:
                coding_helper = self.agent.get_best_helper('coding')
            
            # Enhanced prompt with strategy context
            prompt = f"Create Python code for: {task_description}"
            if parameters.get('approach'):
                prompt += f"\n\nUse this approach: {parameters['approach']}"
            if parameters.get('libraries'):
                prompt += f"\n\nUse these libraries: {', '.join(parameters['libraries'])}"
            
            code = coding_helper.think(prompt)
            
            # Extract and test code
            if '```python' in code:
                start = code.find('```python') + 9
                end = code.find('```', start)
                if end != -1:
                    code = code[start:end].strip()
            
            return self.agent.code_executor.write_and_test_code(
                parameters.get('filename', 'generated_code.py'), 
                code
            )
            
        except Exception as e:
            return {'success': False, 'error': f"Coding strategy execution failed: {e}"}
    
    def execute_file_strategy(self, strategy: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Execute file operation strategy"""
        try:
            parameters = strategy.get('parameters', {})
            
            # File creation with strategy parameters
            filename = parameters.get('filename', 'created_file.txt')
            content = parameters.get('content', f"Content for: {task_description}")
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {
                    'success': True,
                    'action': 'file_created',
                    'filename': filename,
                    'content_length': len(content)
                }
            except Exception as e:
                return {'success': False, 'error': f"File creation failed: {e}"}
                
        except Exception as e:
            return {'success': False, 'error': f"File strategy execution failed: {e}"}
    
    def execute_system_strategy(self, strategy: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Execute system automation strategy"""
        try:
            parameters = strategy.get('parameters', {})
            method = parameters.get('method', 'taskkill')
            
            logger.info(f"🖥️ Executing system strategy: {strategy['name']} using {method}")
            
            if method == 'taskkill':
                return self.execute_taskkill_strategy(parameters, task_description)
            elif method == 'powershell':
                return self.execute_powershell_strategy(parameters, task_description)
            elif method == 'wmi':
                return self.execute_wmi_strategy(parameters, task_description)
            else:
                return {'success': False, 'error': f"Unknown system method: {method}"}
                
        except Exception as e:
            return {'success': False, 'error': f"System strategy execution failed: {e}"}
    
    def execute_taskkill_strategy(self, parameters: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Execute taskkill-based system automation"""
        try:
            # Determine target process from task description
            task_lower = task_description.lower()
            
            if 'notepad' in task_lower:
                process_name = 'notepad.exe'
            elif 'chrome' in task_lower or 'browser' in task_lower:
                process_name = 'chrome.exe'
            elif 'word' in task_lower:
                process_name = 'winword.exe'
            else:
                # Try to extract process name from parameters or task
                process_name = parameters.get('process_name', 'notepad.exe')
            
            # Build taskkill command
            force_flag = '/f' if parameters.get('force', True) else ''
            command = f'taskkill /im {process_name} {force_flag}'
            
            logger.info(f"🔧 Executing: {command}")
            
            # Execute the command
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'action': 'processes_terminated',
                    'process_name': process_name,
                    'command': command,
                    'output': result.stdout.strip(),
                    'message': f"Successfully closed all {process_name} processes"
                }
            else:
                # Check if it's just "no processes found" which is actually success
                if 'not found' in result.stderr.lower() or 'no tasks' in result.stderr.lower():
                    return {
                        'success': True,
                        'action': 'no_processes_found',
                        'process_name': process_name,
                        'message': f"No {process_name} processes were running"
                    }
                else:
                    return {
                        'success': False,
                        'action': 'taskkill_failed',
                        'error': result.stderr.strip(),
                        'command': command
                    }
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Command execution timeout'}
        except Exception as e:
            return {'success': False, 'error': f"Taskkill execution failed: {e}"}
    
    def execute_powershell_strategy(self, parameters: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Execute PowerShell-based system automation"""
        try:
            task_lower = task_description.lower()
            
            # Build PowerShell script based on task
            if 'notepad' in task_lower:
                ps_script = "Get-Process -Name notepad -ErrorAction SilentlyContinue | Stop-Process -Force"
            elif 'close all' in task_lower and 'file' in task_lower:
                ps_script = "Get-Process | Where-Object {$_.ProcessName -like '*notepad*' -or $_.ProcessName -like '*wordpad*'} | Stop-Process -Force"
            else:
                # Generic process termination
                ps_script = parameters.get('script', "Get-Process -Name notepad -ErrorAction SilentlyContinue | Stop-Process -Force")
            
            # Execute PowerShell command
            command = ['powershell', '-Command', ps_script]
            
            logger.info(f"🔧 Executing PowerShell: {ps_script}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'success': result.returncode == 0,
                'action': 'powershell_executed',
                'script': ps_script,
                'output': result.stdout.strip(),
                'error': result.stderr.strip() if result.stderr else None,
                'message': f"PowerShell command executed {'successfully' if result.returncode == 0 else 'with errors'}"
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'PowerShell execution timeout'}
        except Exception as e:
            return {'success': False, 'error': f"PowerShell execution failed: {e}"}
    
    def execute_wmi_strategy(self, parameters: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Execute WMI-based system automation"""
        try:
            task_lower = task_description.lower()
            
            # Build WMI query based on task
            if 'notepad' in task_lower:
                wmi_script = "Get-WmiObject -Query \"SELECT * FROM Win32_Process WHERE Name='notepad.exe'\" | ForEach-Object { $_.Terminate() }"
            else:
                wmi_script = parameters.get('query', "Get-WmiObject -Query \"SELECT * FROM Win32_Process WHERE Name='notepad.exe'\" | ForEach-Object { $_.Terminate() }")
            
            # Execute WMI command through PowerShell
            command = ['powershell', '-Command', wmi_script]
            
            logger.info(f"🔧 Executing WMI: {wmi_script}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'success': result.returncode == 0,
                'action': 'wmi_executed',
                'script': wmi_script,
                'output': result.stdout.strip(),
                'error': result.stderr.strip() if result.stderr else None,
                'message': f"WMI command executed {'successfully' if result.returncode == 0 else 'with errors'}"
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'WMI execution timeout'}
        except Exception as e:
            return {'success': False, 'error': f"WMI execution failed: {e}"}
    
    def execute_deepcoder_strategy(self, strategy: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Execute deepcoder-generated strategy"""
        try:
            # Execute the code/approach provided by deepcoder
            approach = strategy.get('approach', '')
            code = strategy.get('code', '')
            
            if code:
                # Execute the generated code
                return self.agent.code_executor.write_and_test_code('deepcoder_solution.py', code)
            else:
                # Execute the approach description
                return self.agent.process_request(approach)
                
        except Exception as e:
            return {'success': False, 'error': f"Deepcoder strategy execution failed: {e}"}
    
    def generate_emergency_strategy(self, task_description: str, failed_attempts: List[StrategyAttempt]) -> Dict[str, Any]:
        """Generate an emergency strategy when all predefined strategies fail"""
        try:
            # Analyze what failed in previous attempts
            failed_approaches = [attempt.approach for attempt in failed_attempts]
            failure_reasons = [attempt.failure_reason for attempt in failed_attempts if attempt.failure_reason]
            
            # Generate strategy using AI
            strategy_prompt = f"""I need an alternative strategy for this task that has failed multiple times:

TASK: {task_description}

FAILED APPROACHES:
{chr(10).join([f"- {approach}" for approach in failed_approaches])}

FAILURE REASONS:
{chr(10).join([f"- {reason}" for reason in failure_reasons])}

Please provide a completely different approach that avoids these failure modes. Be creative and think outside the box.

Provide your response as:
STRATEGY_NAME: [name]
APPROACH: [detailed approach]
TYPE: [web_automation/coding/file_operation/general]
PARAMETERS: [specific parameters as JSON]
"""
            
            strategy_helper = self.agent.get_best_helper('reasoning')
            if strategy_helper:
                response = strategy_helper.think(strategy_prompt)
                
                # Parse the response
                strategy = self.parse_emergency_strategy(response)
                return strategy
            
            # Fallback strategy
            return {
                'name': 'emergency_fallback',
                'description': 'Basic retry with modified parameters',
                'type': 'general',
                'parameters': {'retry': True, 'modified': True}
            }
            
        except Exception as e:
            logger.error(f"Emergency strategy generation failed: {e}")
            return {
                'name': 'final_fallback',
                'description': 'System fallback approach',
                'type': 'general',
                'parameters': {}
            }
    
    def parse_emergency_strategy(self, response: str) -> Dict[str, Any]:
        """Parse AI-generated emergency strategy"""
        try:
            strategy = {
                'name': 'ai_generated_strategy',
                'description': 'AI-generated alternative approach',
                'type': 'general',
                'parameters': {}
            }
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('STRATEGY_NAME:'):
                    strategy['name'] = line.split(':', 1)[1].strip()
                elif line.startswith('APPROACH:'):
                    strategy['description'] = line.split(':', 1)[1].strip()
                elif line.startswith('TYPE:'):
                    strategy['type'] = line.split(':', 1)[1].strip()
                elif line.startswith('PARAMETERS:'):
                    try:
                        params_str = line.split(':', 1)[1].strip()
                        strategy['parameters'] = json.loads(params_str)
                    except:
                        strategy['parameters'] = {}
            
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to parse emergency strategy: {e}")
            return {
                'name': 'parsed_fallback',
                'description': response[:200],
                'type': 'general',
                'parameters': {}
            }
            
        except Exception as e:
            logger.error(f"Failed to parse emergency strategy: {e}")
            return {
                'name': 'parsed_fallback',
                'description': response[:200],
                'type': 'general',
                'parameters': {}
            }
    
    def consult_deepcoder_for_solution(self, task_description: str, failed_attempts: List[StrategyAttempt]) -> Dict[str, Any]:
        """Consult deepcoder for a solution when all strategies have failed"""
        try:
            # Prepare consultation prompt
            failed_details = []
            for attempt in failed_attempts:
                failed_details.append(f"Strategy: {attempt.strategy_name}")
                failed_details.append(f"Approach: {attempt.approach}")
                failed_details.append(f"Failure: {attempt.failure_reason}")
                failed_details.append("---")
            
            consultation_prompt = f"""I need help with a task that has failed multiple times:

TASK: {task_description}

FAILED ATTEMPTS:
{chr(10).join(failed_details)}

Please provide a completely new approach or solution. This should be different from all the failed attempts.

Format your response as:
SOLUTION_TYPE: [strategy/code/approach]
DESCRIPTION: [detailed description]
IMPLEMENTATION: [specific implementation details or code]
"""
            
            # Get deepcoder interface
            deepcoder_interface = self.agent.capability_gap_analyzer.trigger_capability_development(
                task_description, 
                "Multiple strategy failures", 
                "comprehensive_solution"
            )
            
            if deepcoder_interface.get('capability_developed', False):
                return {
                    'success': True,
                    'strategy': {
                        'name': 'deepcoder_comprehensive_solution',
                        'description': 'Deepcoder-generated comprehensive solution',
                        'type': 'deepcoder_generated',
                        'approach': deepcoder_interface.get('development_result', {}).get('description', ''),
                        'code': deepcoder_interface.get('development_result', {}).get('code', '')
                    }
                }
            else:
                return {'success': False, 'error': 'Deepcoder consultation failed'}
                
        except Exception as e:
            logger.error(f"Deepcoder solution consultation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def learn_from_success(self, strategy: Dict[str, Any], domain: TaskDomain, execution_result: Dict[str, Any]):
        """Learn from successful strategy execution"""
        try:
            # Record successful strategy
            success_record = {
                'strategy': strategy,
                'domain': domain.value,
                'execution_result': execution_result,
                'timestamp': datetime.now().isoformat()
            }
            
            self.strategy_history.append(success_record)
            
            # Update strategy performance weights
            strategy_name = strategy.get('name', 'unknown')
            if strategy_name not in self.performance_weights:
                self.performance_weights[strategy_name] = {'success_count': 0, 'total_attempts': 0}
            
            self.performance_weights[strategy_name]['success_count'] += 1
            self.performance_weights[strategy_name]['total_attempts'] += 1
            
            logger.info(f"🎯 Learned from successful strategy: {strategy_name}")
            
        except Exception as e:
            logger.error(f"Failed to learn from success: {e}")
    
    def generate_persistence_summary(self, attempts: List[StrategyAttempt], goal_achieved: bool) -> str:
        """Generate a summary of the persistence process"""
        try:
            if goal_achieved:
                successful_strategy = next(
                    (attempt for attempt in attempts if attempt.result == StrategyResult.SUCCESS), 
                    None
                )
                if successful_strategy:
                    return f"Goal achieved after {len(attempts)} attempts using '{successful_strategy.strategy_name}' strategy."
                else:
                    return f"Goal achieved after {len(attempts)} attempts."
            else:
                failed_strategies = [attempt.strategy_name for attempt in attempts]
                return f"Goal not achieved after {len(attempts)} attempts. Failed strategies: {', '.join(failed_strategies)}"
                
        except Exception as e:
            return f"Persistence summary generation failed: {e}"
    
    def get_chatgpt_js_script(self) -> str:
        """Get JavaScript for ChatGPT interaction"""
        return """
        // Find and interact with ChatGPT input
        function sendMessageToChatGPT(message) {
            const textarea = document.querySelector('textarea');
            if (textarea) {
                textarea.value = message;
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                
                // Find send button
                const sendButton = document.querySelector('[data-testid="send-button"]') || 
                                 document.querySelector('button[aria-label*="Send"]') ||
                                 textarea.parentNode.querySelector('button');
                
                if (sendButton) {
                    sendButton.click();
                    return 'Message sent successfully';
                } else {
                    // Try pressing Enter
                    textarea.dispatchEvent(new KeyboardEvent('keydown', {
                        key: 'Enter',
                        code: 'Enter',
                        bubbles: true
                    }));
                    return 'Message sent via Enter key';
                }
            }
            return 'Textarea not found';
        }
        
        return sendMessageToChatGPT('Hey');
        """
    
    def save_strategy_history(self):
        """Save strategy history to disk"""
        try:
            # Keep only recent history (last 100 records)
            self.strategy_history = self.strategy_history[-100:]
            
            with open('strategy_history.json', 'w') as f:
                json.dump({
                    'history': self.strategy_history,
                    'performance_weights': getattr(self, 'performance_weights', {})
                }, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save strategy history: {e}")
    
    def load_strategy_history(self):
        """Load strategy history from disk"""
        try:
            if os.path.exists('strategy_history.json'):
                with open('strategy_history.json', 'r') as f:
                    data = json.load(f)
                    self.strategy_history = data.get('history', [])
                    self.performance_weights = data.get('performance_weights', {})
            else:
                self.performance_weights = {}
                
        except Exception as e:
            logger.warning(f"Failed to load strategy history: {e}")
            self.performance_weights = {}

class StrategyEngine:
    """Provides domain-specific strategies for task execution"""
    
    def __init__(self, agent):
        self.agent = agent
        self.strategies = self.initialize_strategies()
    
    def initialize_strategies(self) -> Dict[TaskDomain, List[Dict[str, Any]]]:
        """Initialize predefined strategies for each domain"""
        return {
            TaskDomain.WEB_AUTOMATION: [
                {
                    'name': 'basic_web_automation',
                    'description': 'Basic web automation with standard selectors',
                    'type': 'web_automation',
                    'parameters': {
                        'input_selector': 'textarea',
                        'wait_time': 10,
                        'retry_count': 2
                    }
                },
                {
                    'name': 'advanced_chatgpt_interaction',
                    'description': 'Advanced ChatGPT interaction with multiple selectors',
                    'type': 'web_automation',
                    'parameters': {
                        'input_selector': 'textarea, [contenteditable="true"], input[type="text"]',
                        'response_selector': '[data-message-author-role="assistant"], .markdown, .response-text',
                        'wait_time': 30,
                        'retry_count': 3
                    }
                },
                {
                    'name': 'javascript_web_interaction',
                    'description': 'JavaScript-based web interaction for complex sites',
                    'type': 'web_automation',
                    'parameters': {
                        'use_javascript': True,
                        'js_injection': True,
                        'wait_time': 15
                    }
                }
            ],
            TaskDomain.CODING: [
                {
                    'name': 'simple_code_generation',
                    'description': 'Simple code generation with basic model',
                    'type': 'coding',
                    'parameters': {
                        'model': 'llama3.2:latest',
                        'approach': 'direct',
                        'complexity': 'simple'
                    }
                },
                {
                    'name': 'advanced_code_generation',
                    'description': 'Advanced code generation with specialized model',
                    'type': 'coding',
                    'parameters': {
                        'model': 'deepcoder:latest',
                        'approach': 'systematic',
                        'libraries': ['standard_library'],
                        'complexity': 'advanced'
                    }
                },
                {
                    'name': 'iterative_code_development',
                    'description': 'Iterative code development with testing',
                    'type': 'coding',
                    'parameters': {
                        'model': 'codellama:7b',
                        'approach': 'test_driven',
                        'iterations': 3,
                        'auto_fix': True
                    }
                }
            ],
            TaskDomain.SYSTEM: [
                {
                    'name': 'powershell_automation',
                    'description': 'PowerShell-based system automation',
                    'type': 'system_automation',
                    'parameters': {
                        'method': 'powershell',
                        'admin_required': False
                    }
                },
                {
                    'name': 'taskkill_automation',
                    'description': 'Process termination using taskkill command',
                    'type': 'system_automation',
                    'parameters': {
                        'method': 'taskkill',
                        'force': True,
                        'process_name': 'notepad.exe'
                    }
                },
                {
                    'name': 'wmi_automation',
                    'description': 'Windows Management Interface automation',
                    'type': 'system_automation',
                    'parameters': {
                        'method': 'wmi',
                        'query_based': True
                    }
                }
            ],
            TaskDomain.GENERAL: [
                {
                    'name': 'direct_execution',
                    'description': 'Direct task execution without preprocessing',
                    'type': 'general',
                    'parameters': {
                        'simple': True,
                        'quick': True
                    }
                },
                {
                    'name': 'analyzed_execution',
                    'description': 'Task execution with deep analysis',
                    'type': 'general',
                    'parameters': {
                        'analyze': True,
                        'context_aware': True,
                        'multi_model': True
                    }
                }
            ]
        }
    
    def get_strategies_for_domain(self, domain: TaskDomain) -> List[Dict[str, Any]]:
        """Get strategies for a specific domain"""
        return self.strategies.get(domain, self.strategies[TaskDomain.GENERAL])

class MetaCognition:
    """Meta-cognitive abilities for self-awareness and reflection"""
    
    def __init__(self, agent):
        self.agent = agent
        self.self_model = {
            'current_capabilities': [],
            'learned_skills': [],
            'performance_metrics': {},
            'goals': [],
            'limitations': [],
            'confidence_levels': {},
            'recent_experiences': [],
            'improvement_plan': []
        }
        self.load_self_model()
    
    def reflect_on_experience(self, experience: Dict) -> Dict[str, Any]:
        """Deep reflection on an experience to extract learning"""
        try:
            reflection_prompt = f"""Analyze this experience deeply:

Experience: {json.dumps(experience, indent=2)}

Provide a meta-cognitive analysis:
1. What did I learn about my capabilities?
2. What patterns can I identify?
3. How should I improve my approach?
4. What new capabilities might I need?
5. How confident am I in similar future tasks?
6. What would I do differently?

Be specific and actionable in your analysis.
"""
            
            # Use the best performing helper for reflection
            best_helper = self.agent.get_best_helper('reasoning')
            if best_helper:
                reflection = best_helper.think(reflection_prompt)
                
                # Extract structured insights
                insights = self.extract_insights(reflection)
                
                # Update self-model
                self.update_self_model(insights)
                
                return insights
            
            return {'reflection': 'No suitable helper available for reflection'}
            
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return {'error': str(e)}
    
    def extract_insights(self, reflection: str) -> Dict[str, Any]:
        """Extract structured insights from reflection text"""
        insights = {
            'learned_capabilities': [],
            'identified_patterns': [],
            'improvement_suggestions': [],
            'needed_capabilities': [],
            'confidence_assessment': 0.5,
            'approach_modifications': []
        }
        
        # Simple pattern extraction (could be enhanced with NLP)
        lines = reflection.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'capabilities' in line.lower():
                current_section = 'learned_capabilities'
            elif 'patterns' in line.lower():
                current_section = 'identified_patterns'
            elif 'improve' in line.lower():
                current_section = 'improvement_suggestions'
            elif 'need' in line.lower() and 'capabilit' in line.lower():
                current_section = 'needed_capabilities'
            elif 'confident' in line.lower():
                # Extract confidence level
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    insights['confidence_assessment'] = min(float(numbers[0]) / 100, 1.0)
            elif line.startswith('-') or line.startswith('•'):
                if current_section and current_section in insights:
                    insights[current_section].append(line[1:].strip())
        
        return insights
    
    def update_self_model(self, insights: Dict[str, Any]):
        """Update internal self-model based on insights"""
        if 'learned_capabilities' in insights:
            self.self_model['learned_skills'].extend(insights['learned_capabilities'])
        
        if 'improvement_suggestions' in insights:
            self.self_model['improvement_plan'].extend(insights['improvement_suggestions'])
        
        if 'confidence_assessment' in insights:
            # Update overall confidence
            domain = 'general'
            self.self_model['confidence_levels'][domain] = insights['confidence_assessment']
        
        # Keep only recent experiences (last 50)
        self.self_model['recent_experiences'] = self.self_model['recent_experiences'][-50:]
        
        self.save_self_model()
    
    def save_self_model(self):
        """Save self-model to disk"""
        try:
            with open('self_model.json', 'w') as f:
                json.dump(self.self_model, f, indent=2, cls=CustomJSONEncoder)
        except Exception as e:
            logger.warning(f"Failed to save self-model: {e}")
    
    def load_self_model(self):
        """Load self-model from disk"""
        try:
            if os.path.exists('self_model.json'):
                with open('self_model.json', 'r') as f:
                    saved_model = json.load(f)
                    self.self_model.update(saved_model)
                logger.info("Loaded existing self-model")
        except Exception as e:
            logger.warning(f"Failed to load self-model: {e}")
    
    def assess_capability_for_task(self, task_description: str) -> float:
        """Assess confidence level for a specific task"""
        # Simple keyword matching for now (could be enhanced with embeddings)
        task_lower = task_description.lower()
        
        relevant_confidence = []
        for domain, confidence in self.self_model['confidence_levels'].items():
            if domain in task_lower or any(skill.lower() in task_lower for skill in self.self_model['learned_skills']):
                relevant_confidence.append(confidence)
        
        if relevant_confidence:
            return sum(relevant_confidence) / len(relevant_confidence)
        
        return 0.3  # Default low confidence for unknown tasks

class DynamicCapabilityManager:
    """Manages dynamic capability creation and expansion"""
    
    def __init__(self, agent):
        self.agent = agent
        self.custom_capabilities = {}
        self.load_custom_capabilities()
    
    def create_capability(self, capability_name: str, description: str, implementation_code: str) -> bool:
        """Dynamically create a new capability"""
        try:
            logger.info(f"Creating new capability: {capability_name}")
            
            # Validate and sanitize code
            if not self.validate_code(implementation_code):
                return False
            
            # Create capability file
            capability_file = f"capabilities/{capability_name}.py"
            os.makedirs("capabilities", exist_ok=True)
            
            with open(capability_file, 'w') as f:
                f.write(implementation_code)
            
            # Try to import and test the capability
            try:
                spec = importlib.util.spec_from_file_location(capability_name, capability_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Store capability info
                self.custom_capabilities[capability_name] = {
                    'description': description,
                    'file': capability_file,
                    'module': module,
                    'created': datetime.now().isoformat(),
                    'usage_count': 0
                }
                
                self.save_custom_capabilities()
                logger.info(f"✅ Successfully created capability: {capability_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load capability {capability_name}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create capability: {e}")
            return False
    
    def validate_code(self, code: str) -> bool:
        """Validate code for safety"""
        try:
            # Parse code to check syntax
            ast.parse(code)
            
            # Check for dangerous operations
            dangerous_patterns = [
                'os.system', 'subprocess.call', 'eval(', 'exec(',
                '__import__', 'open(', 'file(', 'input(',
                'raw_input(', 'execfile(', 'compile('
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code:
                    logger.warning(f"Potentially dangerous code detected: {pattern}")
                    return False
            
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return False
    
    def save_custom_capabilities(self):
        """Save custom capabilities metadata"""
        try:
            metadata = {}
            for name, info in self.custom_capabilities.items():
                metadata[name] = {
                    'description': info['description'],
                    'file': info['file'],
                    'created': info['created'],
                    'usage_count': info['usage_count']
                }
            
            with open('custom_capabilities.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save custom capabilities: {e}")
    
    def load_custom_capabilities(self):
        """Load existing custom capabilities"""
        try:
            if os.path.exists('custom_capabilities.json'):
                with open('custom_capabilities.json', 'r') as f:
                    metadata = json.load(f)
                
                for name, info in metadata.items():
                    if os.path.exists(info['file']):
                        try:
                            spec = importlib.util.spec_from_file_location(name, info['file'])
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            self.custom_capabilities[name] = {
                                **info,
                                'module': module
                            }
                            
                        except Exception as e:
                            logger.warning(f"Failed to load capability {name}: {e}")
                
                logger.info(f"Loaded {len(self.custom_capabilities)} custom capabilities")
                
        except Exception as e:
            logger.warning(f"Failed to load custom capabilities: {e}")

class CodeExecutor:
    """Executes and tests code with error handling and debugging"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def write_and_test_code(self, filename: str, code: str, test_instructions: str = None) -> Dict[str, Any]:
        """Write code to file and test it"""
        try:
            logger.info(f"Writing and testing code: {filename}")
            
            # Write code to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Test the code
            test_result = self.test_code(filename, test_instructions)
            
            # If there are errors, try to fix them
            if not test_result['success'] and test_result.get('error'):
                logger.info("Code has errors, attempting to fix...")
                fixed_code = self.debug_and_fix_code(code, test_result['error'])
                
                if fixed_code and fixed_code != code:
                    # Write fixed code
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(fixed_code)
                    
                    # Test again
                    test_result = self.test_code(filename, test_instructions)
                    test_result['was_fixed'] = True
            
            return test_result
            
        except Exception as e:
            logger.error(f"Failed to write and test code: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_code(self, filename: str, test_instructions: str = None) -> Dict[str, Any]:
        """Test code execution"""
        try:
            result = {'success': False, 'output': '', 'error': '', 'execution_time': 0}
            
            start_time = time.time()
            
            if filename.endswith('.py'):
                # Test Python code
                process = subprocess.run(
                    [sys.executable, filename],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                result['output'] = process.stdout
                result['error'] = process.stderr
                result['success'] = process.returncode == 0
                
            else:
                # For other file types, just check if they exist
                result['success'] = os.path.exists(filename)
                result['output'] = f"File {filename} created successfully"
            
            result['execution_time'] = time.time() - start_time
            
            return result
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Code execution timeout', 'output': '', 'execution_time': 30}
        except Exception as e:
            return {'success': False, 'error': str(e), 'output': '', 'execution_time': 0}
    
    def debug_and_fix_code(self, code: str, error_message: str) -> Optional[str]:
        """Use AI to debug and fix code"""
        try:
            debug_prompt = f"""The following Python code has an error:

CODE:
{code}

ERROR:
{error_message}

Please analyze the error and provide the corrected code. Return ONLY the corrected Python code, no explanations.
"""
            
            # Use the best coding helper
            coding_helper = self.agent.get_best_helper('coding')
            if coding_helper:
                fixed_code = coding_helper.think(debug_prompt)
                
                # Extract just the code part
                if '```python' in fixed_code:
                    start = fixed_code.find('```python') + 9
                    end = fixed_code.find('```', start)
                    if end != -1:
                        return fixed_code[start:end].strip()
                elif '```' in fixed_code:
                    start = fixed_code.find('```') + 3
                    end = fixed_code.find('```', start)
                    if end != -1:
                        return fixed_code[start:end].strip()
                
                return fixed_code.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Code debugging failed: {e}")
            return None

class IntelligentPlanner:
    """Creates and executes multi-step plans"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def create_plan(self, goal: str, context: str = "") -> Dict[str, Any]:
        """Create a detailed plan to achieve a goal"""
        try:
            logger.info(f"Creating plan for: {goal}")
            
            planning_prompt = f"""Create a detailed step-by-step plan to achieve this goal:

GOAL: {goal}

CONTEXT: {context}

Available capabilities:
- Code writing and testing
- File operations
- Web browsing
- System automation
- AI model consultation
- Dynamic capability creation

Please create a plan with the following structure:
1. Analysis of the goal
2. Required steps (numbered)
3. Success criteria
4. Potential challenges
5. Risk mitigation

Be specific and actionable. Each step should be something that can be executed.
"""
            
            # Use best planning helper
            planner = self.agent.get_best_helper('planning')
            if planner:
                plan_text = planner.think(planning_prompt)
                
                # Parse plan into structured format
                parsed_plan = self.parse_plan(plan_text)
                parsed_plan['original_text'] = plan_text
                parsed_plan['created'] = datetime.now().isoformat()
                parsed_plan['goal'] = goal
                
                return parsed_plan
            
            return {'error': 'No planning helper available'}
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {'error': str(e)}
    
    def parse_plan(self, plan_text: str) -> Dict[str, Any]:
        """Parse plan text into structured format"""
        plan = {
            'analysis': '',
            'steps': [],
            'success_criteria': [],
            'challenges': [],
            'risk_mitigation': []
        }
        
        current_section = None
        step_counter = 0
        
        for line in plan_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if 'analysis' in line.lower():
                current_section = 'analysis'
            elif 'steps' in line.lower() or 'step' in line.lower():
                current_section = 'steps'
            elif 'success' in line.lower():
                current_section = 'success_criteria'
            elif 'challenge' in line.lower():
                current_section = 'challenges'
            elif 'risk' in line.lower() or 'mitigation' in line.lower():
                current_section = 'risk_mitigation'
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                if current_section == 'steps':
                    step_counter += 1
                    plan['steps'].append({
                        'number': step_counter,
                        'description': line,
                        'completed': False
                    })
            elif line.startswith('-') or line.startswith('•'):
                if current_section and current_section in plan:
                    if current_section == 'steps':
                        step_counter += 1
                        plan['steps'].append({
                            'number': step_counter,
                            'description': line[1:].strip(),
                            'completed': False
                        })
                    else:
                        plan[current_section].append(line[1:].strip())
            elif current_section == 'analysis':
                plan['analysis'] += ' ' + line
        
        return plan
    
    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plan step by step"""
        try:
            logger.info(f"Executing plan for: {plan.get('goal', 'Unknown goal')}")
            
            execution_log = []
            
            for step in plan.get('steps', []):
                if step['completed']:
                    continue
                
                logger.info(f"Executing step {step['number']}: {step['description']}")
                
                step_result = self.execute_step(step['description'])
                step['completed'] = step_result['success']
                
                execution_log.append({
                    'step': step['number'],
                    'description': step['description'],
                    'result': step_result,
                    'timestamp': datetime.now().isoformat()
                })
                
                if not step_result['success']:
                    logger.warning(f"Step {step['number']} failed: {step_result.get('error', 'Unknown error')}")
                    # Try to adapt the plan
                    self.adapt_plan_on_failure(plan, step, step_result)
            
            return {
                'success': all(step['completed'] for step in plan['steps']),
                'execution_log': execution_log,
                'completed_steps': sum(1 for step in plan['steps'] if step['completed']),
                'total_steps': len(plan['steps'])
            }
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute_step(self, step_description: str) -> Dict[str, Any]:
        """Execute a single step"""
        try:
            # Analyze what this step requires
            step_analysis = self.analyze_step(step_description)
            
            # Route to appropriate executor
            if 'code' in step_analysis.get('type', '').lower():
                return self.agent.process_request(step_description)
            elif 'web' in step_analysis.get('type', '').lower():
                return self.agent.process_request(step_description)
            elif 'file' in step_analysis.get('type', '').lower():
                return self.agent.process_request(step_description)
            else:
                # General execution
                return self.agent.process_request(step_description)
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_step(self, step_description: str) -> Dict[str, Any]:
        """Analyze what a step requires"""
        step_lower = step_description.lower()
        
        analysis = {'type': 'general', 'requirements': []}
        
        if any(word in step_lower for word in ['code', 'program', 'script', 'function']):
            analysis['type'] = 'coding'
        elif any(word in step_lower for word in ['web', 'browser', 'website', 'url']):
            analysis['type'] = 'web'
        elif any(word in step_lower for word in ['file', 'create', 'write', 'save']):
            analysis['type'] = 'file'
        elif any(word in step_lower for word in ['open', 'run', 'execute']):
            analysis['type'] = 'system'
        
        return analysis
    
    def adapt_plan_on_failure(self, plan: Dict[str, Any], failed_step: Dict, error_result: Dict):
        """Adapt plan when a step fails"""
        try:
            adaptation_prompt = f"""A plan step failed. Please suggest adaptations:

FAILED STEP: {failed_step['description']}
ERROR: {error_result.get('error', 'Unknown error')}

CURRENT PLAN: {json.dumps(plan['steps'], indent=2)}

How should I adapt this plan to work around the failure? Provide:
1. Modified approach for the failed step
2. Any additional steps needed
3. Changes to subsequent steps
"""
            
            adapter = self.agent.get_best_helper('planning')
            if adapter:
                adaptation = adapter.think(adaptation_prompt)
                logger.info(f"Plan adaptation suggested: {adaptation}")
                
        except Exception as e:
            logger.error(f"Plan adaptation failed: {e}")

class TrulyIntelligentAgent:
    """The main intelligent agent with meta-cognitive abilities"""
    
    def __init__(self):
        self.agent_id = str(uuid.uuid4())[:8]
        self.ollama = ollama.Client()
        
        logger.info(f"🧠 Initializing Truly Intelligent Agent {self.agent_id}")
        
        # Discover available models
        self.available_models = self.discover_models()
        logger.info(f"🤖 Found {len(self.available_models)} available models")
        
        # Initialize intelligent model manager
        self.model_manager = IntelligentModelManager(self.ollama)
        
        # Initialize AI helpers with different specialties
        self.ai_helpers = self.initialize_helpers()
        
        # Initialize meta-cognitive systems
        self.meta_cognition = MetaCognition(self)
        self.capability_manager = DynamicCapabilityManager(self)
        self.code_executor = CodeExecutor(self)
        self.planner = IntelligentPlanner(self)
        
        # Initialize advanced systems
        self.task_analyzer = TaskAnalyzer(self)
        self.capability_gap_analyzer = CapabilityGapAnalyzer(self)
        self.web_master = WebMasterPro(self)
        
        # Initialize goal validation and persistent debugging systems
        self.goal_validator = TaskGoalValidator(self)
        self.strategy_engine = StrategyEngine(self)
        self.persistent_debugger = PersistentDebugger(self)
        
        # Initialize base capabilities
        self.initialize_base_capabilities()
        
        logger.info("✅ Truly Intelligent Agent ready!")
    
    def discover_models(self) -> List[str]:
        """Discover all available AI models"""
        try:
            models = self.ollama.list()
            model_names = []
            for m in models.get('models', []):
                if isinstance(m, dict) and 'name' in m:
                    model_names.append(m['name'])
                elif isinstance(m, str):
                    model_names.append(m)
            return model_names if model_names else ['rolandroland/llama3.1-uncensored:latest']
        except Exception as e:
            logger.warning(f"Model discovery failed: {e}")
            return ['rolandroland/llama3.1-uncensored:latest']  # Fallback
    
    def initialize_helpers(self) -> Dict[str, AIHelper]:
        """Initialize specialized AI helpers"""
        helpers = {}
        
        # Define specialties and assign models
        specialties = {
            'reasoning': 'General reasoning and problem solving',
            'coding': 'Programming, debugging, and software development',
            'planning': 'Strategic planning and task decomposition',
            'analysis': 'Data analysis and pattern recognition',
            'creativity': 'Creative thinking and idea generation',
            'research': 'Information gathering and research',
            'debugging': 'Error analysis and troubleshooting'
        }
        
        # Assign models to specialties (cycling through available models)
        for i, (specialty, description) in enumerate(specialties.items()):
            model_index = i % len(self.available_models)
            model_name = self.available_models[model_index]
            
            helpers[specialty] = AIHelper(model_name, description, self.ollama)
            logger.info(f"📋 Assigned {model_name} to {specialty}")
        
        return helpers
    
    def initialize_base_capabilities(self):
        """Initialize base automation capabilities"""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            self.automation_available = True
            logger.info("✅ Automation capabilities available")
        except ImportError:
            self.automation_available = False
            logger.warning("⚠️ Limited automation - PyAutoGUI not available")
    
    def get_best_helper(self, specialty: str) -> Optional[AIHelper]:
        """Get the best performing helper for a specialty"""
        if specialty in self.ai_helpers:
            return self.ai_helpers[specialty]
        
        # If specific specialty not found, return the highest performing helper
        if self.ai_helpers:
            return max(self.ai_helpers.values(), key=lambda h: h.performance_score)
        
        return None
    
    def summon_helper(self, task_description: str) -> AIHelper:
        """Summon the most appropriate helper for a task"""
        # Analyze task to determine best helper
        task_lower = task_description.lower()
        
        # Map keywords to specialties
        specialty_keywords = {
            'coding': ['code', 'program', 'script', 'debug', 'python', 'function', 'class'],
            'planning': ['plan', 'strategy', 'steps', 'organize', 'schedule'],
            'analysis': ['analyze', 'data', 'pattern', 'statistics', 'research'],
            'creativity': ['creative', 'idea', 'design', 'innovative', 'brainstorm'],
            'debugging': ['error', 'bug', 'fix', 'problem', 'troubleshoot']
        }
        
        # Find best matching specialty
        best_specialty = 'reasoning'  # Default
        max_matches = 0
        
        for specialty, keywords in specialty_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in task_lower)
            if matches > max_matches:
                max_matches = matches
                best_specialty = specialty
        
        helper = self.get_best_helper(best_specialty)
        if helper:
            logger.info(f"🤝 Summoned {helper.model_name} ({best_specialty}) for task")
            return helper
        
        # Fallback: create a new helper with a random available model
        if self.available_models:
            import random
            model = random.choice(self.available_models)
            helper = AIHelper(model, f"General assistant for: {task_description[:50]}", self.ollama)
            logger.info(f"🆘 Created emergency helper: {model}")
            return helper
        
        raise Exception("No AI helpers available")
    
    def think_deeply(self, problem: str, context: str = "") -> Dict[str, Any]:
        """Deep thinking using multiple AI perspectives"""
        try:
            logger.info(f"🤔 Deep thinking about: {problem[:50]}...")
            
            # Get perspectives from multiple helpers
            perspectives = {}
            
            for specialty, helper in self.ai_helpers.items():
                try:
                    perspective = helper.think(problem, context)
                    perspectives[specialty] = perspective
                except Exception as e:
                    logger.warning(f"Helper {specialty} failed: {e}")
            
            # Synthesize perspectives
            synthesis_prompt = f"""Analyze these different perspectives on the problem:

PROBLEM: {problem}

PERSPECTIVES:
{json.dumps(perspectives, indent=2)}

Provide a synthesized analysis that:
1. Identifies key insights from each perspective
2. Finds common themes and contradictions  
3. Recommends the best approach
4. Provides actionable next steps
"""
            
            # Use reasoning helper for synthesis
            synthesizer = self.get_best_helper('reasoning')
            if synthesizer:
                synthesis = synthesizer.think(synthesis_prompt)
                
                return {
                    'problem': problem,
                    'perspectives': perspectives,
                    'synthesis': synthesis,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {'error': 'No synthesizer available', 'perspectives': perspectives}
            
        except Exception as e:
            logger.error(f"Deep thinking failed: {e}")
            return {'error': str(e)}
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process user request with full intelligence like Cline"""
        try:
            logger.info(f"🔄 Processing: {user_input[:50]}...")
            
            # Step 1: Analyze the task comprehensively
            task_analysis = self.task_analyzer.analyze_task(user_input)
            logger.info(f"📊 Task Analysis: {task_analysis.domain.value} | {task_analysis.complexity.value} | Confidence: {task_analysis.confidence:.2f}")
            
            # Step 2: Check if this is an action-oriented request that needs persistent execution
            action_oriented = self.is_action_oriented_request(user_input)
            
            if action_oriented:
                logger.info("🎯 Action-oriented request detected, using persistent debugging...")
                result = self.persistent_debugger.execute_with_persistence(user_input, task_analysis)
                
                # Validate if goal was actually achieved
                validation = self.goal_validator.validate_goal_achievement(user_input, result.get('execution_result', {}))
                
                if not validation['goal_achieved']:
                    logger.warning("❌ Goal not achieved despite execution attempts")
                    result['goal_validation'] = validation
                    result['success'] = False
                else:
                    logger.info("✅ Goal successfully achieved!")
                    result['goal_validation'] = validation
                    result['success'] = True
                
                return result
            
            # Step 3: Select optimal model for this task
            best_model = self.model_manager.get_best_model_for_task(task_analysis)
            if best_model:
                logger.info(f"🎯 Selected optimal model: {best_model}")
            
            # Step 4: Check if we have required capabilities
            missing_capabilities = []
            for capability in task_analysis.required_capabilities:
                if not self.has_capability(capability):
                    missing_capabilities.append(capability)
            
            if missing_capabilities:
                logger.info(f"⚠️ Missing capabilities detected: {missing_capabilities}")
            
            # Step 5: Execute based on complexity and confidence
            start_time = time.time()
            
            try:
                if task_analysis.confidence < 0.3 or missing_capabilities:
                    # Low confidence or missing capabilities - think deeply first
                    logger.info("🤔 Low confidence or missing capabilities, engaging deep analysis...")
                    deep_thought = self.think_deeply(user_input)
                    context = deep_thought.get('synthesis', '')
                    
                    # Try to execute with deep context
                    result = self.execute_with_context(user_input, task_analysis, context, best_model)
                    
                    # If still failed, trigger capability development
                    if not result.get('success', False):
                        gap_analysis = self.capability_gap_analyzer.analyze_failure(
                            user_input, 
                            result.get('error', 'Execution failed'),
                            {'task_analysis': asdict(task_analysis), 'missing_capabilities': missing_capabilities}
                        )
                        
                        if gap_analysis.get('capability_developed', False):
                            logger.info("🚀 New capability developed! Retrying task...")
                            result = self.execute_with_context(user_input, task_analysis, context, best_model)
                
                elif task_analysis.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
                    # Complex task - use planning approach
                    result = self.handle_complex_task(user_input, "")
                else:
                    # Simple/moderate task - direct execution
                    result = self.execute_with_context(user_input, task_analysis, "", best_model)
                
                # Step 5: Update model performance
                execution_time = time.time() - start_time
                success = result.get('success', False)
                
                if best_model:
                    self.model_manager.update_model_performance(best_model, success, execution_time)
                
                # Step 6: Record experience for learning
                experience = {
                    'task': user_input,
                    'task_analysis': asdict(task_analysis),
                    'model_used': best_model,
                    'execution_time': execution_time,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.meta_cognition.self_model['recent_experiences'].append(experience)
                self.meta_cognition.save_self_model()
                
                return result
                
            except Exception as execution_error:
                # Execution failed - analyze gap and potentially develop capability
                logger.error(f"Execution failed: {execution_error}")
                
                gap_analysis = self.capability_gap_analyzer.analyze_failure(
                    user_input,
                    str(execution_error),
                    {'task_analysis': asdict(task_analysis)}
                )
                
                if gap_analysis.get('capability_developed', False):
                    logger.info("🚀 New capability developed after failure! Retrying...")
                    return self.execute_with_context(user_input, task_analysis, "", best_model)
                else:
                    return {
                        'success': False,
                        'error': str(execution_error),
                        'gap_analysis': gap_analysis,
                        'task_analysis': asdict(task_analysis)
                    }
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute_with_context(self, user_input: str, task_analysis: TaskAnalysis, context: str, model: str) -> Dict[str, Any]:
        """Execute task with given context and optimal model"""
        try:
            # Handle web automation tasks
            if task_analysis.domain == TaskDomain.WEB_AUTOMATION:
                return self.handle_web_automation_task(user_input, context)
            
            # Handle coding tasks with best model
            elif task_analysis.domain == TaskDomain.CODING:
                return self.handle_coding_task(user_input, context, model)
            
            # Handle general tasks
            else:
                return self.handle_simple_task(user_input, context)
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def handle_web_automation_task(self, user_input: str, context: str) -> Dict[str, Any]:
        """Handle web automation tasks using WebMasterPro"""
        try:
            logger.info("🌐 Handling web automation task...")
            
            # Parse the web automation request
            if 'chatgpt' in user_input.lower():
                # Example: ChatGPT interaction
                actions = [
                    {'type': 'wait', 'selector': 'textarea', 'selector_type': 'css'},
                    {'type': 'type', 'selector': 'textarea', 'text': 'Hey', 'press_enter': True},
                    {'type': 'wait', 'selector': '[data-message-author-role="assistant"]', 'timeout': 30}
                ]
                
                result = self.web_master.navigate_and_interact('https://chatgpt.com', actions)
                
                if result['success']:
                    # Try to read the response
                    try:
                        # This would need more sophisticated parsing in real implementation
                        message = "ChatGPT responded (screenshot taken for analysis)"
                        return {
                            'success': True,
                            'action': 'web_automation_completed',
                            'message': message,
                            'screenshot': result.get('screenshot'),
                            'details': result
                        }
                    except Exception as e:
                        return {
                            'success': True,
                            'action': 'web_automation_partial',
                            'message': 'Web interaction completed but could not read response',
                            'screenshot': result.get('screenshot'),
                            'error': str(e)
                        }
                else:
                    return {
                        'success': False,
                        'action': 'web_automation_failed',
                        'error': result.get('error', 'Unknown web automation error')
                    }
            
            else:
                # Generic web automation - extract URL and actions from user input
                url_match = re.search(r'(https?://[\w\.-]+\.[\w]+(?:/[\w/.-]*)?)', user_input)
                if not url_match:
                    url_match = re.search(r'([\w\.-]+\.[\w]+)', user_input)
                
                if url_match:
                    url = url_match.group(1)
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    
                    # Basic actions for opening website
                    actions = [{'type': 'screenshot'}]
                    
                    result = self.web_master.navigate_and_interact(url, actions)
                    
                    return {
                        'success': result['success'],
                        'action': 'website_opened',
                        'url': url,
                        'screenshot': result.get('screenshot'),
                        'page_title': result.get('page_title')
                    }
                else:
                    return {'success': False, 'error': 'No valid URL found in web automation request'}
                    
        except Exception as e:
            logger.error(f"Web automation failed: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            # Always close browser after task
            self.web_master.close_browser()
    
    def handle_coding_task(self, user_input: str, context: str, model: str) -> Dict[str, Any]:
        """Handle coding tasks with optimal model selection"""
        try:
            logger.info(f"💻 Handling coding task with model: {model}")
            
            # Use the selected optimal model for code generation
            if model and model in [h.model_name for h in self.ai_helpers.values()]:
                # Find helper with this model
                coding_helper = None
                for helper in self.ai_helpers.values():
                    if helper.model_name == model:
                        coding_helper = helper
                        break
                
                if not coding_helper:
                    coding_helper = self.get_best_helper('coding')
            else:
                coding_helper = self.get_best_helper('coding')
            
            if coding_helper:
                # Generate code with context
                full_prompt = f"Create Python code for: {user_input}"
                if context:
                    full_prompt += f"\n\nAdditional context: {context}"
                
                code = coding_helper.think(full_prompt)
                
                # Extract and test the code
                if '```python' in code:
                    start = code.find('```python') + 9
                    end = code.find('```', start)
                    if end != -1:
                        code = code[start:end].strip()
                elif '```' in code:
                    start = code.find('```') + 3
                    end = code.find('```', start)
                    if end != -1:
                        code = code[start:end].strip()
                
                # Test the code
                result = self.code_executor.write_and_test_code('generated_code.py', code)
                
                return {
                    'success': result['success'],
                    'action': 'code_generated_and_tested',
                    'filename': 'generated_code.py',
                    'code': code,
                    'test_result': result,
                    'model_used': coding_helper.model_name
                }
            else:
                return {'success': False, 'error': 'No coding helper available'}
                
        except Exception as e:
            logger.error(f"Coding task failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability"""
        # Check built-in capabilities
        builtin_capabilities = {
            'web_automation': SELENIUM_AVAILABLE,
            'file_operations': True,
            'code_generation': True,
            'system_automation': self.automation_available,
            'image_processing': False  # Would need additional libraries
        }
        
        if capability in builtin_capabilities:
            return builtin_capabilities[capability]
        
        # Check custom capabilities
        return capability in self.capability_manager.custom_capabilities
    
    def requires_planning(self, user_input: str) -> bool:
        """Determine if a task requires complex planning"""
        planning_keywords = [
            'create', 'build', 'develop', 'make', 'design', 'implement',
            'test', 'debug', 'fix', 'improve', 'optimize', 'analyze'
        ]
        
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in planning_keywords)
    
    def handle_complex_task(self, user_input: str, context: str = "") -> Dict[str, Any]:
        """Handle complex tasks that require planning"""
        try:
            logger.info("🎯 Handling complex task with planning...")
            
            # Create a plan
            plan = self.planner.create_plan(user_input, context)
            
            if 'error' in plan:
                return {'success': False, 'error': plan['error']}
            
            # Execute the plan
            execution_result = self.planner.execute_plan(plan)
            
            # Reflect on the experience
            experience = {
                'task': user_input,
                'plan': plan,
                'execution': execution_result,
                'timestamp': datetime.now().isoformat()
            }
            
            reflection = self.meta_cognition.reflect_on_experience(experience)
            
            return {
                'success': execution_result['success'],
                'plan': plan,
                'execution': execution_result,
                'reflection': reflection,
                'message': f"Complex task completed with {execution_result['completed_steps']}/{execution_result['total_steps']} steps successful"
            }
            
        except Exception as e:
            logger.error(f"Complex task handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_simple_task(self, user_input: str, context: str = "") -> Dict[str, Any]:
        """Handle simple, direct tasks"""
        try:
            logger.info("⚡ Handling simple task...")
            
            # Summon appropriate helper
            helper = self.summon_helper(user_input)
            
            # Get response from helper
            response = helper.think(user_input, context)
            
            # Try to execute if it's an actionable task
            action_result = self.try_execute_action(user_input, response)
            
            # Update helper performance
            helper.update_performance(action_result.get('success', True))
            
            return {
                'success': True,
                'response': response,
                'action_result': action_result,
                'helper_used': f"{helper.model_name} ({helper.specialty})"
            }
            
        except Exception as e:
            logger.error(f"Simple task handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def try_execute_action(self, user_input: str, ai_response: str) -> Dict[str, Any]:
        """Try to execute an action based on user input"""
        try:
            user_lower = user_input.lower()
            
            # File creation
            if 'create' in user_lower and 'file' in user_lower:
                filename_match = re.search(r'called|named\s+(\S+)', user_input)
                filename = filename_match.group(1) if filename_match else 'new_file.txt'
                
                content_match = re.search(r'with\s+(.+)', user_input)
                content = content_match.group(1) if content_match else ai_response[:200]
                
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return {'success': True, 'action': 'file_created', 'filename': filename}
                except Exception as e:
                    return {'success': False, 'action': 'file_creation_failed', 'error': str(e)}
            
            # Code creation and testing
            elif any(word in user_lower for word in ['code', 'script', 'program']):
                if 'snake' in user_lower and 'game' in user_lower:
                    return self.create_snake_game()
                else:
                    # Generic code creation
                    code_helper = self.get_best_helper('coding')
                    if code_helper:
                        code = code_helper.think(f"Create Python code for: {user_input}")
                        return self.code_executor.write_and_test_code('generated_code.py', code)
            
            # Web browsing
            elif any(word in user_lower for word in ['open', 'go to', 'visit']) and any(url in user_lower for url in ['.com', '.org', '.net']):
                url_match = re.search(r'([\w\.-]+\.[\w]+)', user_input)
                if url_match:
                    url = url_match.group(1)
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    
                    try:
                        webbrowser.open(url)
                        return {'success': True, 'action': 'website_opened', 'url': url}
                    except Exception as e:
                        return {'success': False, 'action': 'website_open_failed', 'error': str(e)}
            
            # System automation
            elif self.automation_available and any(word in user_lower for word in ['open', 'type', 'click']):
                return self.execute_automation(user_input)
            
            # No specific action detected
            return {'success': True, 'action': 'response_only', 'message': 'Provided intelligent response'}
            
        except Exception as e:
            return {'success': False, 'action': 'execution_failed', 'error': str(e)}
    
    def create_snake_game(self) -> Dict[str, Any]:
        """Create a working Snake game"""
        try:
            logger.info("🐍 Creating Snake game...")
            
            snake_code = '''
import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Game settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // CELL_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // CELL_SIZE

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

class Snake:
    def __init__(self):
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (0, -1)  # Moving up initially
        self.grow = False
    
    def move(self):
        head_x, head_y = self.positions[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        # Check boundaries
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            return False
        
        # Check self collision
        if new_head in self.positions:
            return False
        
        self.positions.insert(0, new_head)
        
        if not self.grow:
            self.positions.pop()
        else:
            self.grow = False
        
        return True
    
    def change_direction(self, direction):
        # Prevent moving backwards
        if (direction[0] * -1, direction[1] * -1) != self.direction:
            self.direction = direction
    
    def eat_food(self):
        self.grow = True
    
    def draw(self, screen):
        for position in self.positions:
            rect = pygame.Rect(position[0] * CELL_SIZE, position[1] * CELL_SIZE, 
                             CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, rect)

class Food:
    def __init__(self):
        self.position = self.randomize_position()
    
    def randomize_position(self):
        return (random.randint(0, GRID_WIDTH - 1), 
                random.randint(0, GRID_HEIGHT - 1))
    
    def draw(self, screen):
        rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE,
                          CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, rect)

def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake Game")
    clock = pygame.time.Clock()
    
    snake = Snake()
    food = Food()
    score = 0
    font = pygame.font.Font(None, 36)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.change_direction((0, -1))
                elif event.key == pygame.K_DOWN:
                    snake.change_direction((0, 1))
                elif event.key == pygame.K_LEFT:
                    snake.change_direction((-1, 0))
                elif event.key == pygame.K_RIGHT:
                    snake.change_direction((1, 0))
        
        # Move snake
        if not snake.move():
            # Game over
            game_over_text = font.render(f"Game Over! Score: {score}", True, WHITE)
            screen.blit(game_over_text, (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2))
            pygame.display.flip()
            pygame.time.wait(3000)
            pygame.quit()
            sys.exit()
        
        # Check food collision
        if snake.positions[0] == food.position:
            snake.eat_food()
            food.position = food.randomize_position()
            # Make sure food doesn't spawn on snake
            while food.position in snake.positions:
                food.position = food.randomize_position()
            score += 1
        
        # Draw everything
        screen.fill(BLACK)
        snake.draw(screen)
        food.draw(screen)
        
        # Draw score
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(10)  # 10 FPS

if __name__ == "__main__":
    main()
'''
            
            # Write and test the game
            result = self.code_executor.write_and_test_code('snake_game.py', snake_code)
            
            if result['success']:
                logger.info("✅ Snake game created successfully!")
                return {
                    'success': True,
                    'action': 'snake_game_created',
                    'filename': 'snake_game.py',
                    'message': 'Snake game created! Run with: python snake_game.py (requires pygame: pip install pygame)'
                }
            else:
                logger.info("Game creation failed, trying to fix...")
                return {
                    'success': False,
                    'action': 'snake_game_creation_failed',
                    'error': result.get('error', 'Unknown error'),
                    'attempted_fix': result.get('was_fixed', False)
                }
            
        except Exception as e:
            return {'success': False, 'action': 'snake_game_creation_error', 'error': str(e)}
    
    def execute_automation(self, user_input: str) -> Dict[str, Any]:
        """Execute system automation tasks"""
        try:
            if not self.automation_available:
                return {'success': False, 'error': 'PyAutoGUI not available'}
            
            import pyautogui
            user_lower = user_input.lower()
            
            # Open program and type
            if 'open' in user_lower and 'write' in user_lower:
                program_match = re.search(r'open\s+(\w+)', user_lower)
                text_match = re.search(r'write\s+(.+)', user_input)
                
                if program_match and text_match:
                    program = program_match.group(1)
                    text = text_match.group(1)
                    
                    # Open program via Run dialog
                    pyautogui.hotkey('win', 'r')
                    time.sleep(0.5)
                    pyautogui.write(program)
                    pyautogui.press('enter')
                    time.sleep(2)
                    pyautogui.write(text)
                    
                    return {'success': True, 'action': 'automation_completed', 
                           'details': f'Opened {program} and typed: {text}'}
            
            # Just type text
            elif 'type' in user_lower or 'write' in user_lower:
                text_match = re.search(r'(?:type|write)\s+(.+)', user_input)
                if text_match:
                    text = text_match.group(1)
                    pyautogui.write(text)
                    return {'success': True, 'action': 'text_typed', 'text': text}
            
            return {'success': False, 'error': 'Automation command not recognized'}
            
        except Exception as e:
            return {'success': False, 'action': 'automation_failed', 'error': str(e)}
    
    def is_action_oriented_request(self, user_input: str) -> bool:
        """Determine if a request is action-oriented and needs actual execution"""
        user_lower = user_input.lower()
        
        # Action verbs that indicate execution required
        action_verbs = [
            'do', 'execute', 'run', 'perform', 'make', 'create', 'build',
            'close', 'open', 'start', 'stop', 'kill', 'terminate', 'end',
            'send', 'type', 'click', 'press', 'write', 'save', 'delete',
            'install', 'download', 'upload', 'move', 'copy', 'remove',
            'navigate', 'go', 'visit', 'browse', 'search', 'find'
        ]
        
        # Check for action verbs
        has_action_verb = any(verb in user_lower for verb in action_verbs)
        
        # Check for imperative phrases
        imperative_phrases = [
            'do it', 'make it', 'execute it', 'run it', 'perform this',
            'close all', 'open all', 'kill all', 'terminate all',
            'yourself', 'automatically', 'for me', 'please do'
        ]
        
        has_imperative = any(phrase in user_lower for phrase in imperative_phrases)
        
        # Check for system/automation requests
        system_targets = [
            'notepad', 'file', 'program', 'application', 'process',
            'window', 'browser', 'website', 'system', 'computer'
        ]
        
        has_system_target = any(target in user_lower for target in system_targets)
        
        # Check for explicit rejection of explanations
        anti_explanation = [
            'instead of', 'don\'t tell me', 'don\'t explain', 'don\'t give me steps',
            'stop explaining', 'just do', 'actually do', 'yourself'
        ]
        
        wants_action_not_explanation = any(phrase in user_lower for phrase in anti_explanation)
        
        # Action-oriented if it has action verbs + targets, or explicit action request
        return (has_action_verb and has_system_target) or has_imperative or wants_action_not_explanation
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        helper_status = {}
        for name, helper in self.ai_helpers.items():
            helper_status[name] = {
                'model': helper.model_name,
                'performance': helper.performance_score,
                'usage_count': helper.usage_count
            }
        
        return {
            'agent_id': self.agent_id,
            'available_models': self.available_models,
            'ai_helpers': helper_status,
            'automation_available': self.automation_available,
            'custom_capabilities': len(self.capability_manager.custom_capabilities),
            'self_model': self.meta_cognition.self_model,
            'capabilities': [
                'deep_thinking', 'planning', 'code_creation', 'debugging',
                'file_operations', 'web_browsing', 'system_automation',
                'self_reflection', 'capability_expansion'
            ]
        }
    
    def install_dependency(self, package: str) -> Dict[str, Any]:
        """Install a Python package"""
        try:
            logger.info(f"Installing {package}...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                return {'success': True, 'message': f'Successfully installed {package}'}
            else:
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

def main():
    """Main function"""
    print("🧠 Initializing Truly Intelligent Self-Evolving Agent...")
    print("=" * 70)
    
    try:
        agent = TrulyIntelligentAgent()
        
        print(f"✅ Agent initialized!")
        print(f"🆔 ID: {agent.agent_id}")
        print(f"🤖 Available Models: {len(agent.available_models)}")
        print(f"🧬 AI Helpers: {len(agent.ai_helpers)}")
        print(f"🔧 Automation: {'Available' if agent.automation_available else 'Limited'}")
        print("=" * 70)
        
        print("\n🧠 I am a truly intelligent agent capable of:")
        print("   • Deep thinking with multiple AI perspectives")
        print("   • Complex planning and execution")
        print("   • Self-reflection and learning")
        print("   • Dynamic capability expansion")
        print("   • Code creation, testing, and debugging")
        print("   • System automation and web interaction")
        print("   • Meta-cognitive self-awareness")
        print("\n   Commands: 'status', 'helpers', 'reflect', 'install <package>', 'quit'")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\n🗣️  ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("\n👋 Farewell! I have evolved through our interaction.")
                    break
                
                elif user_input.lower() == 'status':
                    print("\n📊 Comprehensive Agent Status:")
                    status = agent.get_status()
                    for key, value in status.items():
                        if isinstance(value, dict):
                            print(f"   {key}:")
                            for sub_key, sub_value in value.items():
                                print(f"     {sub_key}: {sub_value}")
                        else:
                            print(f"   {key}: {value}")
                
                elif user_input.lower() == 'helpers':
                    print("\n🤖 AI Helper Status:")
                    for name, helper in agent.ai_helpers.items():
                        print(f"   {name}: {helper.model_name} (Performance: {helper.performance_score:.2f}, Uses: {helper.usage_count})")
                
                elif user_input.lower() == 'reflect':
                    print("\n🧠 Engaging in self-reflection...")
                    reflection = agent.meta_cognition.reflect_on_experience({
                        'type': 'self_reflection',
                        'timestamp': datetime.now().isoformat(),
                        'current_state': agent.get_status()
                    })
                    print(f"Reflection insights: {json.dumps(reflection, indent=2)}")
                
                elif user_input.lower().startswith('install '):
                    package = user_input[8:].strip()
                    print(f"\n🔧 Installing {package}...")
                    result = agent.install_dependency(package)
                    print(f"   {result}")
                
                else:
                    # Process the request with full intelligence
                    print(f"\n🧠 Processing with full intelligence: {user_input}")
                    result = agent.process_request(user_input)
                    
                    print(f"\n📋 Result:")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if key != 'plan' and key != 'execution':  # Skip complex nested data
                                print(f"   {key}: {value}")
                    else:
                        print(f"   {result}")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Main loop error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

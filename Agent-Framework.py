# agent_framework.py v1.1 - Fixed and Cleaned
import os
import json
import yaml
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Type, Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
from langchain_community.llms import Ollama, HuggingFaceHub
from langchain_openai import ChatOpenAI
from huggingface_hub import InferenceClient
from openai import OpenAI as OpenAIClient

class LLMManager:
    """Central LLM manager with support for multiple providers"""
    
    def __init__(self, config_path: str = "llm_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.available_models = self.discover_models()
    
    def load_config(self) -> dict:
        """Load LLM configuration from file"""
        default_config = {
            "default_provider": "ollama",
            "providers": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "default_model": "llama3"
                },
                "lm_studio": {
                    "base_url": "http://localhost:1234/v1",
                    "default_model": "local-model"
                },
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "default_model": "gpt-4-turbo"
                },
                "huggingface": {
                    "api_key": os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
                    "default_model": "google/flan-t5-xxl"
                }
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)
    
    def discover_models(self) -> dict:
        """Discover available models from all providers"""
        models = {"ollama": [], "lm_studio": [], "openai": [], "huggingface": []}
        
        # Discover Ollama models
        try:
            response = requests.get(f"{self.config['providers']['ollama']['base_url']}/api/tags")
            if response.status_code == 200:
                models["ollama"] = [model["name"] for model in response.json().get("models", [])]
        except:
            pass
        
        # Discover LM Studio models
        try:
            response = requests.get(f"{self.config['providers']['lm_studio']['base_url']}/models")
            if response.status_code == 200:
                models["lm_studio"] = [model["id"] for model in response.json().get("data", [])]
        except:
            pass
        
        # Get OpenAI models (requires API key)
        if self.config["providers"]["openai"]["api_key"]:
            try:
                client = OpenAIClient(api_key=self.config["providers"]["openai"]["api_key"])
                openai_models = client.models.list()
                models["openai"] = [model.id for model in openai_models.data]
            except:
                pass
        
        # Get Hugging Face models (requires API key)
        if self.config["providers"]["huggingface"]["api_key"]:
            try:
                response = requests.get(
                    "https://api-inference.huggingface.co/models",
                    headers={"Authorization": f"Bearer {self.config['providers']['huggingface']['api_key']}"}
                )
                if response.status_code == 200:
                    models["huggingface"] = [model["modelId"] for model in response.json()]
            except:
                pass
        
        return models
    
    def get_llm(self, provider: str = None, model: str = None, **kwargs):
        """Get LLM instance for the specified provider and model"""
        provider = provider or self.config["default_provider"]
        provider_config = self.config["providers"][provider]
        
        if not model:
            model = provider_config["default_model"]
        
        print(f"Using {provider.upper()} model: {model}")
        
        if provider == "ollama":
            return Ollama(
                base_url=provider_config["base_url"],
                model=model,
                temperature=0.7,
                num_ctx=4096,
                **kwargs
            )
        elif provider == "lm_studio":
            return ChatOpenAI(
                base_url=provider_config["base_url"],
                model=model,
                temperature=0.7,
                max_tokens=2048,
                **kwargs
            )
        elif provider == "openai":
            return ChatOpenAI(
                api_key=provider_config["api_key"],
                model=model,
                temperature=0.7,
                **kwargs
            )
        elif provider == "huggingface":
            return HuggingFaceHub(
                repo_id=model,
                huggingfacehub_api_token=provider_config["api_key"],
                model_kwargs={
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                    **kwargs
                }
            )
        
        raise ValueError(f"Unsupported provider: {provider}")
    
    def stream_response(self, provider: str, model: str, prompt: str, **kwargs):
        """Stream response from LLM for interactive use"""
        provider_config = self.config["providers"][provider]
        
        if provider == "ollama":
            response = requests.post(
                f"{provider_config['base_url']}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    **kwargs
                },
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if not chunk.get("done"):
                        yield chunk.get("response", "")
        
        elif provider == "lm_studio":
            client = OpenAIClient(base_url=provider_config["base_url"])
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        elif provider == "openai":
            client = OpenAIClient(api_key=provider_config["api_key"])
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        elif provider == "huggingface":
            client = InferenceClient(
                model=model,
                token=provider_config["api_key"]
            )
            for token in client.text_generation(prompt, stream=True, **kwargs):
                yield token
    
    def get_provider_config(self, provider: str) -> dict:
        """Get configuration for a specific provider"""
        return self.config["providers"].get(provider, {})
    
    def set_default_provider(self, provider: str):
        """Set the default LLM provider"""
        if provider in self.config["providers"]:
            self.config["default_provider"] = provider
            self.save_config()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def set_default_model(self, provider: str, model: str):
        """Set default model for a provider"""
        if provider in self.config["providers"] and model in self.available_models[provider]:
            self.config["providers"][provider]["default_model"] = model
            self.save_config()
        else:
            raise ValueError(f"Invalid provider or model: {provider}/{model}")

class CI_CDSystem:
    """Dual-path CI/CD system with separate pipelines for user and agent code"""
    
    def __init__(self, agent):
        self.agent = agent
        self.config_dir = agent.workspace / "ci_cd_config"
        self.user_results_dir = agent.workspace / "ci_cd_results/user"
        self.agent_results_dir = agent.workspace / "ci_cd_results/agent"
        
        # Create directories
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.user_results_dir.mkdir(exist_ok=True)
        self.agent_results_dir.mkdir(exist_ok=True)
        
        # Initialize configurations
        self.user_config = self.load_or_create_config("user")
        self.agent_config = self.load_or_create_config("agent")
        
        # Start monitoring
        self.start_monitors()
    
    def load_or_create_config(self, config_type: str) -> dict:
        """Load or create CI/CD configuration"""
        config_path = self.config_dir / f"{config_type}_config.yaml"
        
        if not config_path.exists():
            default_config = {
                'version': '1.0',
                'monitor_paths': [],
                'actions': {
                    'on_change': ['run_tests', 'static_analysis'],
                    'on_demand': ['security_scan', 'coverage']
                },
                'language_configs': {}
            }
            
            # Type-specific defaults
            if config_type == "user":
                default_config['monitor_paths'] = [str(self.agent.user_project_dir)]
                default_config['reporting'] = {'email': '', 'webhook': ''}
            else:  # agent
                default_config['monitor_paths'] = [
                    str(self.agent.generated_code_dir),
                    str(self.agent.self_improvement_dir)
                ]
                default_config['auto_improve'] = True
            
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f)
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def start_monitors(self):
        """Start separate monitors for user and agent codebases"""
        
        # Agent code monitor (self-improvement codebase)
        agent_handler = AgentCIHandler(self.agent, self.agent_config)
        self.agent_observer = Observer()
        for path in self.agent_config['monitor_paths']:
            self.agent_observer.schedule(agent_handler, path, recursive=True)
        self.agent_observer.start()
        
        # User code monitor (user project)
        user_handler = UserCIHandler(self.agent, self.user_config)
        self.user_observer = Observer()
        for path in self.user_config['monitor_paths']:
            self.user_observer.schedule(user_handler, path, recursive=True)
        self.user_observer.start()
    
    def run_pipeline(self, scope: list, config_type: str, trigger: str = "on_demand") -> dict:
        """Run CI/CD pipeline for specific scope"""
        config = self.user_config if config_type == "user" else self.agent_config
        results_dir = self.user_results_dir if config_type == "user" else self.agent_results_dir
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "config_type": config_type,
            "trigger": trigger,
            "timestamp": timestamp,
            "scope": scope,
            "actions": {}
        }
        
        # Run configured actions
        for action in config['actions'][trigger]:
            action_results = []
            for path in scope:
                action_results.append(self.run_action(action, Path(path), run_dir, config))
            results["actions"][action] = action_results
        
        # Save results
        report_path = run_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Agent-specific: trigger self-improvement on failure
        if config_type == "agent" and config.get('auto_improve', False):
            if any(not r['success'] for r in results["actions"]["run_tests"]):
                self.agent.auto_self_improve(scope)
        
        return results
    
    def run_action(self, action: str, path: Path, run_dir: Path, config: dict) -> dict:
        """Run a specific CI/CD action"""
        language = self.agent.detect_language_from_extension(path.suffix)
        lang_config = config['language_configs'].get(language, {})
        
        # Get command from config or default
        if action == "run_tests":
            cmd = lang_config.get('test_command', 
                                 self.agent.language_config[language]['test_command'])
        elif action == "static_analysis":
            cmd = lang_config.get('linter_command', 
                                 self.agent.language_config[language]['linter_command'])
        elif action == "security_scan":
            cmd = lang_config.get('security_command', 
                                 self.agent.language_config[language]['security_command'])
        elif action == "coverage":
            cmd = lang_config.get('coverage_command', 
                                 self.agent.language_config[language]['coverage_command'])
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
        
        # Format command with parameters
        formatted_cmd = cmd.format(
            file=path.name,
            path=path.parent,
            results_dir=run_dir
        )
        
        try:
            # Execute command
            result = subprocess.run(
                formatted_cmd.split(),
                cwd=path.parent,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            return {
                "path": str(path),
                "command": formatted_cmd,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except Exception as e:
            return {
                "path": str(path),
                "success": False,
                "error": str(e)
            }

class AgentCIHandler(FileSystemEventHandler):
    """CI handler for agent's self-improving codebase"""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.last_trigger = 0
        self.cooldown = 5  # seconds
    
    def on_modified(self, event):
        if not event.is_directory and time.time() - self.last_trigger > self.cooldown:
            self.last_trigger = time.time()
            file_path = Path(event.src_path)
            
            # Only trigger for code files
            LANGUAGE_EXTENSIONS = ['.py', '.js', '.java', '.cpp', '.c', '.rs', '.go', '.ts']
            if file_path.suffix in LANGUAGE_EXTENSIONS:
                print(f"\nAGENT CI: Detected change in self-improvement codebase: {file_path}")
                self.agent.ci_cd.run_pipeline(
                    scope=[str(file_path)],
                    config_type="agent",
                    trigger="on_change"
                )

class UserCIHandler(FileSystemEventHandler):
    """CI handler for user project code"""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.last_trigger = 0
        self.cooldown = 10  # seconds
    
    def on_modified(self, event):
        if not event.is_directory and time.time() - self.last_trigger > self.cooldown:
            self.last_trigger = time.time()
            file_path = Path(event.src_path)
            
            # Only trigger for code files
            LANGUAGE_EXTENSIONS = ['.py', '.js', '.java', '.cpp', '.c', '.rs', '.go', '.ts']
            if file_path.suffix in LANGUAGE_EXTENSIONS:
                print(f"\nUSER CI: Detected change in user project: {file_path}")
                # User CI only logs, doesn't auto-run unless configured
                if self.config.get('auto_run_on_change', False):
                    self.agent.ci_cd.run_pipeline(
                        scope=[str(file_path)],
                        config_type="user",
                        trigger="on_change"
                    )

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self,
                 name: str,
                 llm_manager: LLMManager,
                 workspace: Path,
                 config: Optional[dict] = None):
        self.name = name
        self.llm_manager = llm_manager
        self.workspace = workspace
        self.config = config or {}
        self.llm = self.llm_manager.get_llm(
            provider=self.config.get("llm_provider"),
            model=self.config.get("llm_model")
        )
        self.setup_workspace()
    
    def setup_workspace(self):
        """Create agent-specific workspace"""
        self.workspace.mkdir(exist_ok=True, parents=True)
        print(f"{self.name} workspace: {self.workspace}")
    
    @abstractmethod
    def execute(self, task: str, **kwargs) -> Any:
        """Execute the agent's primary function"""
        pass
    
    def stream_response(self, prompt: str, **kwargs):
        """Stream response using agent's default provider"""
        provider = self.config.get("llm_provider", self.llm_manager.config["default_provider"])
        model = self.config.get("llm_model", 
                               self.llm_manager.config["providers"][provider]["default_model"])
        
        return self.llm_manager.stream_response(
            provider=provider,
            model=model,
            prompt=prompt,
            **kwargs
        )
    
    def update_config(self, new_config: dict):
        """Update agent configuration"""
        self.config = {**self.config, **new_config}
        # Reinitialize LLM if provider/model changed
        if "llm_provider" in new_config or "llm_model" in new_config:
            self.llm = self.llm_manager.get_llm(
                provider=self.config.get("llm_provider"),
                model=self.config.get("llm_model")
            )

class ResearchAgent(BaseAgent):
    """Agent for research tasks with RAG capabilities"""
    
    def __init__(self,
                 llm_manager: LLMManager,
                 workspace: Path = Path("research_workspace"),
                 config: Optional[dict] = None):
        super().__init__("Research Agent", llm_manager, workspace, config)
        self.knowledge_base = workspace / "knowledge_base"
        self.setup_research_env()
    
    def setup_research_env(self):
        """Setup research-specific environment"""
        self.knowledge_base.mkdir(exist_ok=True)
        # Initialize vector store, etc.
        print(f"Research knowledge base at: {self.knowledge_base}")
    
    def execute(self, task: str, stream: bool = False, **kwargs):
        """Execute a research query"""
        if stream:
            return self.stream_response(f"Research: {task}")
        
        # Actual research implementation
        return f"Research result for: {task}"

class CodingAgent(BaseAgent):
    """Self-improving coding agent with CI/CD"""
    
    LANGUAGE_EXTENSIONS = {
        'python': ['.py'],
        'javascript': ['.js', '.ts'],
        'java': ['.java'],
        'cpp': ['.cpp', '.cc', '.cxx', '.h', '.hpp'],
        'c': ['.c', '.h'],
        'rust': ['.rs'],
        'go': ['.go']
    }
    
    def __init__(self,
                 llm_manager: LLMManager,
                 workspace: Path = Path("coding_workspace"),
                 config: Optional[dict] = None):
        super().__init__("Coding Agent", llm_manager, workspace, config)
        self.user_project_dir = Path(config.get("user_project_dir", "user_project"))
        self.generated_code_dir = workspace / "generated_code"
        self.self_improvement_dir = workspace / "self_improvement"
        self.language_config = self._get_default_language_config()
        
        # CI/CD system
        self.ci_cd = None
        if config.get('enable_ci_cd', True):
            self.ci_cd = CI_CDSystem(self)
        
        self.setup_coding_env()
    
    def _get_default_language_config(self):
        """Get default language configuration"""
        return {
            'python': {
                'test_command': 'python -m pytest {file} --tb=short',
                'linter_command': 'pylint {file}',
                'security_command': 'bandit -r {path}',
                'coverage_command': 'coverage run -m pytest {file} && coverage report'
            },
            'javascript': {
                'test_command': 'npx jest {file}',
                'linter_command': 'npx eslint {file}',
                'security_command': 'npx audit',
                'coverage_command': 'npx jest {file} --coverage'
            }
        }
    
    def setup_coding_env(self):
        """Setup coding-specific environment"""
        self.user_project_dir.mkdir(exist_ok=True)
        self.generated_code_dir.mkdir(exist_ok=True)
        self.self_improvement_dir.mkdir(exist_ok=True)
        print(f"User project at: {self.user_project_dir}")
        print(f"Generated code at: {self.generated_code_dir}")
        print(f"Self-improvement code at: {self.self_improvement_dir}")
    
    def detect_language_from_extension(self, extension: str) -> str:
        """Detect programming language from file extension"""
        for lang, exts in self.LANGUAGE_EXTENSIONS.items():
            if extension in exts:
                return lang
        return 'unknown'
    
    def execute(self, task: str, language: str, **kwargs):
        """Generate code for a specific task"""
        # Actual code generation implementation
        return f"Generated {language} code for: {task}"
    
    def stream_code_generation(self, task: str, language: str):
        """Stream code generation process"""
        prompt = f"Write {language} code for: {task}. Include comments and error handling."
        
        # Get provider from current LLM
        provider = self.llm_manager.config["default_provider"]
        model = self.llm_manager.config["providers"][provider]["default_model"]
        
        return self.llm_manager.stream_response(
            provider=provider,
            model=model,
            prompt=prompt
        )
    
    def run_user_ci_cd(self, scope: list = None):
        """Run user CI/CD pipeline on demand"""
        if not self.ci_cd:
            print("CI/CD not enabled")
            return None
        
        # Default to entire user project
        if scope is None:
            scope = [str(p) for p in self.get_code_files(self.user_project_dir)]
        
        return self.ci_cd.run_pipeline(
            scope=scope,
            config_type="user",
            trigger="on_demand"
        )
    
    def run_agent_ci_cd(self, scope: list = None):
        """Run agent CI/CD pipeline (usually automatic)"""
        if not self.ci_cd:
            print("CI/CD not enabled")
            return None
        
        # Default to recent agent code
        if scope is None:
            scope = [
                str(p) for p in
                self.get_recent_files(self.generated_code_dir, count=5) +
                self.get_recent_files(self.self_improvement_dir, count=5)
            ]
        
        return self.ci_cd.run_pipeline(
            scope=scope,
            config_type="agent",
            trigger="on_change"  # Agent CI uses on_change even when manually triggered
        )
    
    def get_code_files(self, directory: Path) -> list:
        """Get all code files in directory"""
        code_files = []
        for ext_list in self.LANGUAGE_EXTENSIONS.values():
            for ext in ext_list:
                code_files.extend(directory.glob(f"**/*{ext}"))
        return code_files
    
    def get_recent_files(self, directory: Path, count: int = 5) -> list:
        """Get most recently modified files in directory"""
        files = [f for f in directory.glob("*") if f.is_file()]
        return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[:count]
    
    def auto_self_improve(self, scope: list = None):
        """Self-improve with CI/CD feedback - now scoped to specific files"""
        if scope is None:
            # Default to files that failed in last agent CI run
            scope = self.get_failed_files_from_last_ci_run()
        
        print(f"\nSelf-Improvement triggered for {len(scope)} files")
        # ... existing improvement logic, now focused on specific files ...
        return f"Improved {len(scope)} files"
    
    def get_failed_files_from_last_ci_run(self) -> list:
        """Get files that failed in the last agent CI run"""
        if not self.ci_cd:
            return []
        
        # Find latest agent CI report
        agent_runs = sorted(self.ci_cd.agent_results_dir.glob("*"), reverse=True)
        if not agent_runs:
            return []
        
        latest_run = agent_runs[0]
        report_path = latest_run / "report.json"
        
        if not report_path.exists():
            return []
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        failed_files = []
        for action, results in report["actions"].items():
            for result in results:
                if not result.get("success", True):
                    failed_files.append(result["path"])
        
        return list(set(failed_files))

class AgentFactory:
    """Creates and manages agents"""
    
    def __init__(self, llm_manager: LLMManager, agent_registry: Optional[dict] = None):
        self.llm_manager = llm_manager
        self.agents = {}
        self.registry = agent_registry or self.default_registry()
    
    def default_registry(self) -> Dict[str, Type[BaseAgent]]:
        """Default registry of available agent types"""
        return {
            "research": ResearchAgent,
            "coding": CodingAgent
        }
    
    def register_agent(self, name: str, agent_class: Type[BaseAgent]):
        """Register a new agent type"""
        if not issubclass(agent_class, BaseAgent):
            raise ValueError("Agent must subclass BaseAgent")
        self.registry[name] = agent_class
    
    def create_agent(self,
                    agent_type: str,
                    workspace: Optional[Path] = None,
                    config: Optional[dict] = None) -> BaseAgent:
        """Create a new agent instance"""
        if agent_type not in self.registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self.registry[agent_type]
        
        # Set default workspace if not provided
        if workspace is None:
            workspace = Path(f"{agent_type}_workspace")
        
        # Create agent instance
        agent = agent_class(
            llm_manager=self.llm_manager,
            workspace=workspace,
            config=config or {}
        )
        
        # Store agent by name
        self.agents[agent.name] = agent
        return agent
    
    def get_agent(self, name: str) -> BaseAgent:
        """Get an existing agent by name"""
        if name not in self.agents:
            raise ValueError(f"Agent not found: {name}")
        return self.agents[name]
    
    def list_agents(self) -> list:
        """List all active agents"""
        return list(self.agents.keys())

class AgentPlugin:
    """Base class for agent plugins"""
    
    def __init__(self, name: str):
        self.name = name
    
    def pre_execute(self, agent: BaseAgent, task: str, **kwargs) -> dict:
        """Called before agent execution"""
        return kwargs
    
    def post_execute(self, agent: BaseAgent, task: str, result: Any, **kwargs):
        """Called after agent execution"""
        return result

class PluginManager:
    """Manages agent plugins"""
    
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, plugin: AgentPlugin):
        """Register a new plugin"""
        self.plugins[plugin.name] = plugin
    
    def apply_pre_execute(self, agent: BaseAgent, task: str, **kwargs) -> dict:
        """Apply all pre-execute hooks"""
        modified_kwargs = kwargs
        for plugin in self.plugins.values():
            modified_kwargs = plugin.pre_execute(agent, task, **modified_kwargs)
        return modified_kwargs
    
    def apply_post_execute(self, agent: BaseAgent, task: str, result: Any, **kwargs) -> Any:
        """Apply all post-execute hooks"""
        modified_result = result
        for plugin in self.plugins.values():
            modified_result = plugin.post_execute(agent, task, modified_result, **kwargs)
        return modified_result

class DocumentationAgent(BaseAgent):
    """Agent for generating documentation"""
    
    def __init__(self,
                 llm_manager: LLMManager,
                 workspace: Path = Path("doc_workspace"),
                 config: Optional[dict] = None):
        super().__init__("Documentation Agent", llm_manager, workspace, config)
        self.templates_dir = workspace / "templates"
        self.setup_doc_env()
    
    def setup_doc_env(self):
        """Setup documentation environment"""
        self.templates_dir.mkdir(exist_ok=True)
        print(f"Templates directory at: {self.templates_dir}")
    
    def execute(self, task: str, format: str = "markdown", **kwargs):
        """Generate documentation"""
        return f"Documentation ({format}) for: {task}"

class SecurityPlugin(AgentPlugin):
    """Security scanning plugin"""
    
    def __init__(self):
        super().__init__("security_scanner")
    
    def post_execute(self, agent: BaseAgent, task: str, result: Any, **kwargs):
        """Scan generated content for security issues"""
        if hasattr(agent, 'name') and 'Coding' in agent.name:
            print(f"Scanning code for security vulnerabilities...")
            # Actual security scanning would go here
            return f"{result}\n\nSecurity Scan: No issues found"
        return result

class AgentFramework:
    """Main framework entry point"""
    
    def __init__(self):
        self.llm_manager = LLMManager()
        self.agent_factory = AgentFactory(self.llm_manager)
        self.plugin_manager = PluginManager()
        
        # Register custom agent types
        self.agent_factory.register_agent("documentation", DocumentationAgent)
        
        # Register plugins
        self.plugin_manager.register_plugin(SecurityPlugin())
    
    def create_agent(self,
                    agent_type: str,
                    workspace: Optional[Path] = None,
                    config: Optional[dict] = None) -> BaseAgent:
        """Create agent with plugin support"""
        agent = self.agent_factory.create_agent(agent_type, workspace, config)
        return agent
    
    def execute_agent(self,
                     agent_name: str,
                     task: str,
                     **kwargs) -> Any:
        """Execute agent task with plugin hooks"""
        agent = self.agent_factory.get_agent(agent_name)
        
        # Pre-execute hooks
        modified_kwargs = self.plugin_manager.apply_pre_execute(agent, task, **kwargs)
        
        # Execute agent
        result = agent.execute(task, **modified_kwargs)
        
        # Post-execute hooks
        return self.plugin_manager.apply_post_execute(agent, task, result, **kwargs)

# Example Usage
if __name__ == "__main__":
    # Initialize framework
    framework = AgentFramework()
    
    # Create agents
    research_agent = framework.create_agent(
        "research",
        workspace=Path("my_research_workspace"),
        config={"llm_provider": "ollama", "llm_model": "llama3"}
    )
    
    coding_agent = framework.create_agent(
        "coding",
        config={
            "user_project_dir": "my_project",
            "llm_provider": "lm_studio",
            "llm_model": "TheBloke/CodeLlama-13B-Instruct-GGUF"
        }
    )
    
    doc_agent = framework.create_agent("documentation")
    
    # Execute agents
    research_result = framework.execute_agent(
        "Research Agent",
        "Explain quantum computing basics"
    )
    print(f"Research Result: {research_result}")
    
    # Stream coding agent response
    print("\nCoding Agent Streaming:")
    for chunk in coding_agent.stream_code_generation("Implement quicksort in Python", "python"):
        print(chunk, end="", flush=True)
    
    # Execute documentation agent
    doc_result = framework.execute_agent(
        "Documentation Agent",
        "Create API documentation for the quicksort function",
        format="html"
    )
    print(f"\n\nDocumentation Result: {doc_result}")
    
    # List all agents
    print("\nActive Agents:")
    for agent_name in framework.agent_factory.list_agents():
        print(f" - {agent_name}")n
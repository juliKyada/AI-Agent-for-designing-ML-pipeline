"""
LLM Client for MetaFlow - Handles communication with Groq models
"""
import os
import json
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv
from src.utils import get_logger, get_config

# Load environment variables
load_dotenv()

logger = get_logger()
config = get_config()

class LLMClient:
    """Client for interacting with Groq LLM models"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq client
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment. LLM features will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            
        self.model = config.get('llm.model', 'llama-3.3-70b-versatile')
        logger.info(f"LLM Client initialized with model: {self.model}")

    def get_completion(self, prompt: str, system_prompt: str = "You are an expert Data Scientist and ML Engineer.") -> str:
        """Get a completion from the LLM"""
        if not self.client:
            return "Error: LLM client not initialized"
            
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=config.get('llm.temperature', 0.1),
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: {str(e)}"

    def get_json_completion(self, prompt: str, system_prompt: str = "You are an expert Data Scientist. Respond only with valid JSON.") -> Dict[str, Any]:
        """Get a JSON response from the LLM"""
        content = self.get_completion(prompt, system_prompt)
        
        try:
            # Find JSON in the response if there's surrounding text
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                return json.loads(json_str)
            else:
                return {"error": "No JSON found in response", "raw_content": content}
        except Exception as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {"error": str(e), "raw_content": content}

    def suggest_pipelines(self, metadata: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
        """
        Suggest ML pipelines based on dataset metadata
        """
        prompt = f"""
        Given the following dataset metadata for a {task_type} task, suggest 3-5 machine learning pipelines.
        For each pipeline, provide:
        1. A descriptive name
        2. The model class (from common libraries like sklearn, xgboost, lightgbm)
        3. Recommended hyperparameter search space
        4. Brief rationale
        
        Dataset Metadata:
        {json.dumps(metadata, indent=2)}
        
        Respond ONLY with a JSON list of objects:
        [
          {{
            "name": "Pipeline Name",
            "model_type": "RandomForestClassifier",
            "library": "sklearn.ensemble",
            "hyperparameters": {{"n_estimators": [100, 200], "max_depth": [null, 10, 20]}},
            "rationale": "Explanation why this model fits"
          }}
        ]
        """
        
        response = self.get_json_completion(prompt)
        if isinstance(response, list):
            return response
        elif isinstance(response, dict) and "pipelines" in response:
            return response["pipelines"]
        elif isinstance(response, dict) and "error" not in response:
            # Probably a single object or list wrapped in an object
            return [response]
        else:
            logger.warning(f"Unexpected LLM response format: {response}")
            return []

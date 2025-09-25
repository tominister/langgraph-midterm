import os
import requests
from .config import LLM_ENDPOINT, LLM_API_KEY

# Optional provider selector: 'openai' to use OpenAI Chat Completions API
LLM_PROVIDER = os.getenv('LLM_PROVIDER', '')


class LLMClient:
    def __init__(self, endpoint: str = None, api_key: str = None):
        self.endpoint = endpoint or LLM_ENDPOINT
        self.api_key = api_key or LLM_API_KEY

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generic HTTP call to an LLM endpoint. You should configure endpoint and key in .env."""
        if not self.endpoint:
            return "[LLM endpoint not configured] This is a stubbed response. Provide a real LLM_ENDPOINT to get model answers."

        # OpenAI provider support (Chat Completions)
        if LLM_PROVIDER.lower() == 'openai':
            # LLM_ENDPOINT should be the base URL, e.g. https://api.openai.com/v1
            url = self.endpoint.rstrip('/') + '/chat/completions'
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            body = {
                'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': max_tokens,
                'temperature': float(os.getenv('LLM_TEMPERATURE', '0.0'))
            }
            resp = requests.post(url, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # extract text from choices
            try:
                return data['choices'][0]['message']['content'].strip()
            except Exception:
                return str(data)

        # Groq provider support: expect the user to set LLM_ENDPOINT to the model's inference URL
        # and LLM_API_KEY to their Groq API key. Groq endpoints vary; the body below is a safe,
        # commonly-accepted shape ({"input": prompt, "max_output_tokens": ...}).
        if LLM_PROVIDER.lower() == 'groq':
            url = self.endpoint.rstrip('/')
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            body = {
                'input': prompt,
                'temperature': float(os.getenv('LLM_TEMPERATURE', '0.0')),
                'max_output_tokens': max_tokens,
            }
            resp = requests.post(url, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Groq responses may vary; try a few common shapes defensively
            if isinstance(data, dict):
                if 'output' in data and isinstance(data['output'], str):
                    return data['output'].strip()
                if 'text' in data:
                    return data['text']
                if 'result' in data:
                    return data['result']
                if 'outputs' in data and isinstance(data['outputs'], list) and data['outputs']:
                    out0 = data['outputs'][0]
                    if isinstance(out0, dict) and 'content' in out0:
                        return out0['content'] if isinstance(out0['content'], str) else str(out0['content'])
                    return str(out0)
            return str(data)

        # Klangoo provider support: POST {"text": prompt} to the analyze endpoint
        if LLM_PROVIDER.lower() == 'klangoo':
            # Default to the provided endpoint or Klangoo's analyze path
            url = self.endpoint.rstrip('/') if self.endpoint else 'https://api.klangoo.com/v1/analyze'
            headers = {
                'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
                'Content-Type': 'application/json'
            }
            body = {'text': prompt}
            resp = requests.post(url, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Defensive parsing: try common keys that may contain textual analysis
            if isinstance(data, dict):
                for key in ('analysis', 'result', 'text', 'output', 'data'):
                    val = data.get(key)
                    if isinstance(val, str):
                        return val.strip()
                    if isinstance(val, dict) and isinstance(val.get('text'), str):
                        return val.get('text').strip()
                # check list shapes
                if 'outputs' in data and isinstance(data['outputs'], list) and data['outputs']:
                    out0 = data['outputs'][0]
                    if isinstance(out0, str):
                        return out0.strip()
                    if isinstance(out0, dict):
                        for k in ('text', 'content', 'result'):
                            if isinstance(out0.get(k), str):
                                return out0.get(k).strip()
            return str(data)

        # Generic HTTP provider: expect JSON with top-level text/result keys
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
        payload = {
            'prompt': prompt,
            'max_tokens': max_tokens,
        }
        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get('text') or data.get('result') or str(data)

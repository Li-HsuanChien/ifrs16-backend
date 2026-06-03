import json
import requests
from typing import List, Dict, Any, Optional

class LLMCLIENT:
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 model: str,
                 headers: Optional[Dict[str, str]] = None
            ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.headers = headers or {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            json_schema: Optional[Dict[str, Any]] = None,
            temperature: float = 0.7,
        )-> Dict[str, Any]:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                
            }
            if json_schema:
                payload["response_format"] = {
                    "type": "json_object",
                    "json_object": {
                        "name": "schema",
                        "schema": json_schema
                    }
                }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            

def extraction(client: LLMCLIENT, system_prompt: str, user_prompt: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        result = client.generate(system_prompt, user_prompt, json_schema)
        return result
    except Exception as e:
        print(f"Error during extraction: {e}")
        return {}
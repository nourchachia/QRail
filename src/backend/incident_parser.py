"""
Incident Parser - src/backend/incident_parser.py

Purpose:
    Parses natural language incident descriptions using Gemini AI to extract:
    - estimated_delay (minutes)
    - primary_failure_code (standardized code)
    - Enriches with weather/load data from live_status.json

Features:
    - Gemini prompt engineering for precise extraction
    - Auto-fetches weather/load from StorageManager.load_json('live_status.json')
    - Structured output parsing
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.backend.database import StorageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Gemini (optional dependency)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "google-generativeai not installed. "
        "Install with: pip install google-generativeai"
    )


class IncidentParser:
    """
    Parses incident descriptions using Gemini AI to extract structured data.
    
    Extracts:
    - estimated_delay: Delay in minutes
    - primary_failure_code: Standardized failure code
    
    Enriches with:
    - weather: From live_status.json
    - network_load_pct: From live_status.json
    """
    
    def __init__(
        self, 
        data_dir: str = "data",
        api_key: Optional[str] = None
    ):
        """Initialize incident parser."""
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        self.storage = StorageManager(data_dir=data_dir)
        
        # Configure Gemini
        if self.api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                logger.info("GenAI configured successfully")
            except Exception as e:
                logger.error(f"Failed to configure GenAI: {e}")
    
    def _load_live_status(self) -> Dict[str, Any]:
        """Load live_status.json to get weather and network load"""
        try:
            live_status = self.storage.load_json("live_status.json")
            if not live_status:
                raise ValueError("live_status.json is empty or missing")
            
            # Extract weather and load strictly - NO DEFAULTS / GUESSES
            weather = live_status.get("weather", {})
            network_load = live_status.get("network_load_pct") # default None
            
            return {
                "weather_condition": weather.get("condition"),
                "temperature_c": weather.get("temperature_c"),
                "wind_speed_kmh": weather.get("wind_speed_kmh"),
                "visibility_km": weather.get("visibility_km"),
                "network_load_pct": network_load,
                "active_trains": live_status.get("active_trains", []),
                "data_source": "Live status sensors (data/processed/live_status.json)"
            }
        except Exception as e:
            logger.warning(f"Failed to load live_status.json: {e}")
            return {
                "weather_condition": None,
                "temperature_c": None,
                "wind_speed_kmh": None,
                "visibility_km": None,
                "network_load_pct": None,
                "active_trains": [],
                "data_source": "VERIFICATION FAILED: Data missing from database (No guess allowed)"
            }
    
    def _create_prompt(self, description: str, context: Dict[str, Any]) -> str:
        """Create Gemini prompt with live telemetry context."""
        
        # Format active train data for the prompt
        trains_summary = ""
        for t in context.get('active_trains', []):
            pos = t['cur_pos']
            loc = pos.get('station_id') or pos.get('segment')
            trains_summary += f"- {t['train_id']} at {loc} ({t['cur_delay']}m delay)\n"
            
        prompt = f"""You are a railway incident analysis system. 
Directly use the LIVE NETWORK STATE below to identify which specific trains are affected.

INCIDENT DESCRIPTION:
{description}

CURRENT NETWORK CONTEXT:
- Weather: {context['weather_condition']}
- Network Load: {context['network_load_pct']}%
- Visibility: {context['visibility_km']} km

LIVE NETWORK STATE (Active Trains):
{trains_summary or "No active trains."}

REQUIRED OUTPUT (JSON format):
{{
    "estimated_delay_minutes": <integer, 0-300>,
    "primary_failure_code": "<standardized code>",
    "station_names": ["list", "of", "mentioned", "stations"],
    "train_id": "<ID of the most affected train from the list above>",
    "confidence": <float, 0.0-1.0>,
    "reasoning": "<explanation identifying specific affected trains from the state above>"
}}

FAILURE CODE STANDARDS:
- SIGNAL_FAIL: Signal system failure
- TRAIN_BREAKDOWN: Train mechanical failure
- PASSENGER_ALARM: Passenger emergency alarm
- WEATHER_SEVERE: Severe weather conditions
- INFRASTRUCTURE_FAULT: Track/infrastructure fault
- POWER_OUTAGE: Electrical power failure
- SWITCH_FAILURE: Point/switch mechanism failure
- COMMUNICATION_LOSS: Communication system failure
- UNKNOWN: Unable to determine

DELAY ESTIMATION GUIDELINES:
- Signal failure: 15-60 minutes (depending on severity)
- Train breakdown: 20-90 minutes (depending on rescue time)
- Passenger alarm: 5-30 minutes (depending on response)
- Severe weather: 10-120 minutes (depending on conditions)
- Infrastructure fault: 30-180 minutes (depending on repair complexity)
- Power outage: 45-240 minutes (depending on backup systems)

Consider the network load and weather conditions when estimating delays.
Higher network load and adverse weather increase delay times.

Output ONLY valid JSON, no additional text."""

        return prompt
    
    def parse(
        self, 
        description: str,
        use_gemini: bool = True
    ) -> Dict[str, Any]:
        """
        Parse incident description to extract structured data.
        
        Args:
            description: Natural language incident description
            use_gemini: Whether to use Gemini AI (False for fallback parsing)
        
        Returns:
            Dictionary with:
            - estimated_delay_minutes: int
            - primary_failure_code: str
            - confidence: float
            - reasoning: str
            - weather: dict (from live_status.json)
            - network_load_pct: int (from live_status.json)
        """
        # Load context from live_status.json
        context = self._load_live_status()
        
        if use_gemini and GEMINI_AVAILABLE:
            try:
                return self._parse_with_gemini(description, context)
            except Exception as e:
                logger.error(f"Gemini parsing failed: {e}")
                raise RuntimeError(f"AI Reasoning Unavailable. Please provide details manually. (Error: {str(e)})")
        else:
            raise RuntimeError("AI infrastructure unavailable and strictly evidence-based mode enabled (Guesses disabled).")
    
    def _parse_with_gemini(
        self, 
        description: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse using Gemini AI"""
        import google.generativeai as genai
        
        # Create the prompt 🧠
        prompt = self._create_prompt(description, context)
        
        # Try multiple models in order of preference (prioritizing proven working models)
        models_to_try = [
            'gemini-flash-latest',      # PROVEN WORKING ✅
            'gemini-2.0-flash-lite-preview-02-05',
            'gemini-2.0-flash',         # Hit rate limits
            'gemini-1.5-flash'
        ]
        response = None
        last_error = None
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                print(f"   ✓ Used Model: {model_name}")
                break # Success!
            except Exception as e:
                last_error = e
                # Don't log warning for expected 404s on fallback trial
                continue
                
        if not response:
            raise last_error or Exception("All Gemini models failed")
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Try to extract JSON if wrapped in markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            return self._parse_fallback(description, context)
        
        # Add context data
        result["weather"] = {
            "condition": context["weather_condition"],
            "temperature_c": context["temperature_c"],
            "wind_speed_kmh": context["wind_speed_kmh"],
            "visibility_km": context["visibility_km"]
        }
        result["network_load_pct"] = context["network_load_pct"]
        
        return result
    
    # LEGACY FALLBACK DELETED - ZERO GUESSING POLICY ENFORCED
    
    def parse_incident(
        self, 
        incident: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse incident dictionary (extracts description and enriches).
        
        Args:
            incident: Incident dictionary with 'semantic_description' or 'description'
        
        Returns:
            Enriched incident with parsed fields
        """
        description = (
            incident.get("semantic_description") or 
            incident.get("description") or 
            ""
        )
        
        parsed = self.parse(description)
        
        # Merge parsed data into incident
        enriched = incident.copy()
        enriched.update({
            "estimated_delay_minutes": parsed["estimated_delay_minutes"],
            "primary_failure_code": parsed["primary_failure_code"],
            "parsing_confidence": parsed["confidence"],
            "parsing_reasoning": parsed["reasoning"]
        })
        
        # Add weather/load if not already present
        if "weather" not in enriched:
            enriched["weather"] = parsed["weather"]
        if "network_load_pct" not in enriched:
            enriched["network_load_pct"] = parsed["network_load_pct"]
        
        return enriched


# Example usage
if __name__ == "__main__":
    """
    Test script for IncidentParser
    
    Expected Output:
    ================
    Parsed incident with estimated_delay_minutes and primary_failure_code
    """
    
    # Initialize parser
    parser = IncidentParser(data_dir="data")
    
    # Test with description
    description = "Signal system failure at Central Station during peak hours. Multiple converging routes affected."
    
    result = parser.parse(description)
    print("Parsed Result:")
    print(json.dumps(result, indent=2))
    
    # Test with incident dictionary
    test_incident = {
        "incident_id": "INC_001",
        "semantic_description": description,
        "type": "signal_failure"
    }
    
    enriched = parser.parse_incident(test_incident)
    print("\nEnriched Incident:")
    print(json.dumps(enriched, indent=2))

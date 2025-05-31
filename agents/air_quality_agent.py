import json
from typing import Dict, Generator
from agents.air_quality_tools import OpenMeteoTools
from llm import llm
import re
import numpy as np
import pandas as pd

class AirQualityAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = OpenMeteoTools()

    def __call__(self, state: Dict) -> Dict:
        user_query = state.get("user_query", "")
        location_input = state.get("location_input", None)
        if location_input:
            user_query = location_input  # Use the provided location directly
        prompt = f"""
<|start_of_role|>system<|end_of_role|>You are a location extractor for the Valley Air chatbot. Given a user query, extract the city, county, or zip code mentioned. Output a single JSON object with keys 'city', 'county', and 'zip'. If not present, leave the value as an empty string. Output ONLY the JSON object, with NO Markdown formatting, NO code blocks, NO explanation, and NO examples.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Query: {user_query}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
"""
        response = self.llm.invoke(prompt)
        print(f"DEBUG: Raw LLM location extraction output: {response!r}")
        # Remove Markdown code block if present
        if response.strip().startswith('```'):
            response = re.sub(r'^```[a-zA-Z]*\n|```$', '', response.strip(), flags=re.MULTILINE).strip()
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = response  # fallback
        try:
            location_info = json.loads(json_str)
            location_str = location_info.get("city") or location_info.get("county") or location_info.get("zip")
            if not location_str:
                return {**state, "needs_location": True}
            geocoded = self.tools.geocode_location(location_str)
            print(f"DEBUG: Geocoded location: {geocoded}")
            if geocoded and self.tools.validate_location(geocoded):
                air_quality = self.tools.get_air_quality(
                    geocoded["latitude"],
                    geocoded["longitude"]
                )
                # Compose the actual API URL for sources (no longer used in sources)
                # api_url = (
                #     f"https://air-quality-api.open-meteo.com/v1/air-quality?"
                #     f"latitude={geocoded['latitude']}&longitude={geocoded['longitude']}"
                #     f"&hourly=pm10,pm2_5,nitrogen_dioxide,carbon_dioxide,ozone,sulphur_dioxide,carbon_monoxide,dust&timezone=auto"
                # )
                if air_quality:
                    summary = air_quality["summary"]
                    timeseries = air_quality["timeseries"]
                    summary_prompt = f"""
<|start_of_role|>system<|end_of_role|>You are an air quality assistant for the Valley Air chatbot. Summarize the following air quality data in a clear, user-friendly way for residents of California's Central Valley. Explain technical terms simply. Output only the answer, with no extra text or formatting.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Location: {{location}}
AQI: {{aqi}} ({{category}})
PM2.5: {{pm2_5}} µg/m³
Ozone: {{ozone}} ppb
Other pollutants: {{other}}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
"""
                    other = ", ".join([
                        f"NO2: {summary['no2']} ppb",
                        f"SO2: {summary['so2']} ppb",
                        f"CO: {summary['co']} ppm"
                    ])
                    summary_text = self.llm.invoke(summary_prompt.format(
                        location=geocoded["name"],
                        aqi=summary["aqi"],
                        category=summary["aqi_category"],
                        pm2_5=summary["pm2_5"],
                        ozone=summary["ozone"],
                        other=other
                    ))
                    return {
                        **state,
                        "answer": summary_text,
                        "air_quality_data": summary,
                        "air_quality_timeseries": timeseries,
                        "location": geocoded,
                        "sources": []  # No API URL in sources
                    }
            elif geocoded:
                # Out of area
                return {**state, "needs_location": True, "location_error": f"Sorry, {location_str} is not in the San Joaquin Valley. Please enter a city, county, or zip code within the valley."}
            return {**state, "needs_location": True}
        except Exception as e:
            print(f"Error in AirQualityAgent: {e}")
            return {**state, "needs_location": True}

    def stream(self, state: Dict, callback_handler=None) -> Generator[Dict, None, None]:
        if callback_handler:
            callback_handler.on_tool_start("AirQualityAgent", "Processing air quality query...")
        result = self.__call__(state)
        if result.get("needs_location"):
            message = result.get("location_error") or "Please enter a city, county, or zip code in the San Joaquin Valley:"
            yield {"type": "location_needed", "message": message}
        else:
            # Yield the air quality time series for charting
            timeseries = result.get("air_quality_timeseries")
            if timeseries:
                cleaned_timeseries = {}
                for k, v in timeseries.items():
                    if k == "time":
                        cleaned_timeseries[k] = v
                    else:
                        cleaned_timeseries[k] = [float(x) if x is not None else np.nan for x in v]
                df = pd.DataFrame(cleaned_timeseries)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    df = df.set_index("time")
                yield {"type": "air_quality", "data": df}
            # Then yield the synthesized answer (no API URL in sources)
            sources = result.get("sources")
            yield {"type": "answer", "content": result.get("answer"), "sources": sources} 
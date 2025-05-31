import os
import json
from typing import Dict, Optional
import requests
import pandas as pd

class OpenMeteoTools:
    def __init__(self):
        self.san_joaquin_valley_locations = {
            "cities": [
                "Fresno", "Bakersfield", "Clovis", "Modesto", "Stockton", "Visalia",
                "Atwater", "Ceres", "Corcoran", "Delano", "Dinuba", "Galt", "Hanford", "Lathrop", "Lemoore", "Lodi", "Los Banos", "Madera", "Manteca", "Merced", "Oakdale", "Patterson", "Porterville", "Reedley", "Riverbank", "Sanger", "Selma", "Shafter", "Tracy", "Tulare", "Turlock", "Wasco",
                "Arvin", "Avenal", "Chowchilla", "Coalinga", "Dos Palos", "Escalon", "Exeter", "Farmersville", "Firebaugh", "Fowler", "Gustine", "Hughson", "Kerman", "Kettleman City", "Keyes", "Kingsburg", "Lindsay", "Livingston", "McFarland", "Mendota", "Newman", "Orange Cove", "Parlier", "Ripon", "San Joaquin", "Taft", "Waterford", "Woodlake"
            ],
            "counties": [
                "Fresno County", "Kern County", "Kings County", "Madera County", "Merced County", "San Joaquin County", "Stanislaus County", "Tulare County"
            ]
        }

    def geocode_location(self, location: str) -> Optional[Dict]:
        try:
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("results"):
                result = data["results"][0]
                return {
                    "name": result.get("name"),
                    "latitude": result.get("latitude"),
                    "longitude": result.get("longitude"),
                    "elevation": result.get("elevation"),
                    "timezone": result.get("timezone"),
                    "country": result.get("country"),
                    "admin1": result.get("admin1"),
                    "admin2": result.get("admin2"),
                    "admin3": result.get("admin3"),
                }
        except Exception as e:
            print(f"Geocoding error: {e}")
        return None

    def get_air_quality(self, latitude: float, longitude: float) -> Optional[Dict]:
        try:
            url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "pm10,pm2_5,nitrogen_dioxide,carbon_dioxide,ozone,sulphur_dioxide,carbon_monoxide,dust",
                "timezone": "auto"
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            if hourly and "time" in hourly:
                # Use pandas to align with previous logic
                times = pd.to_datetime(hourly["time"])
                latest_idx = -1
                def get_latest(var):
                    arr = hourly.get(var, [])
                    for val in reversed(arr):
                        try:
                            if val is not None:
                                return float(val)
                        except Exception:
                            continue
                    return None
                aqi = self._calculate_aqi(get_latest("pm2_5"), get_latest("ozone"))
                summary = {
                    "timestamp": str(times[latest_idx]),
                    "aqi": aqi,
                    "pm2_5": get_latest("pm2_5"),
                    "pm10": get_latest("pm10"),
                    "ozone": get_latest("ozone"),
                    "no2": get_latest("nitrogen_dioxide"),
                    "so2": get_latest("sulphur_dioxide"),
                    "co": get_latest("carbon_monoxide"),
                    "co2": get_latest("carbon_dioxide"),
                    "dust": get_latest("dust"),
                    "aqi_category": self._get_aqi_category(aqi)
                }
                # Return both summary and full timeseries
                return {
                    "summary": summary,
                    "timeseries": hourly
                }
        except Exception as e:
            print(f"Air quality data error: {e}")
        return None

    def _calculate_aqi(self, pm2_5: float, ozone: float) -> int:
        if pm2_5 is None:
            pm2_5_aqi = 0
        elif pm2_5 <= 12.0:
            pm2_5_aqi = 50 * (pm2_5 / 12.0)
        elif pm2_5 <= 35.4:
            pm2_5_aqi = 51 + 49 * ((pm2_5 - 12.1) / (35.4 - 12.1))
        elif pm2_5 <= 55.4:
            pm2_5_aqi = 101 + 49 * ((pm2_5 - 35.5) / (55.4 - 35.5))
        else:
            pm2_5_aqi = 151 + 99 * ((pm2_5 - 55.5) / (150.4 - 55.5))
        if ozone is None:
            ozone_aqi = 0
        elif ozone <= 54:
            ozone_aqi = 50 * (ozone / 54)
        elif ozone <= 70:
            ozone_aqi = 51 + 49 * ((ozone - 55) / (70 - 55))
        elif ozone <= 85:
            ozone_aqi = 101 + 49 * ((ozone - 71) / (85 - 71))
        else:
            ozone_aqi = 151 + 99 * ((ozone - 86) / (105 - 86))
        return max(int(pm2_5_aqi), int(ozone_aqi))

    def _get_aqi_category(self, aqi: int) -> str:
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def validate_location(self, location: Dict) -> bool:
        city = (location.get("admin3") or "").strip()
        county = (location.get("admin2") or "").strip()
        name = (location.get("name") or "").strip()
        # Accept city match in cities list
        if city and city in self.san_joaquin_valley_locations["cities"]:
            return True
        if name and name in self.san_joaquin_valley_locations["cities"]:
            return True
        # Accept county match, with or without ' County'
        for county_name in self.san_joaquin_valley_locations["counties"]:
            if county and (county == county_name or county + ' County' == county_name or county_name.startswith(county)):
                return True
        return False 
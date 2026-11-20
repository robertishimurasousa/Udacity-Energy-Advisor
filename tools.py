"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import json
import random
from datetime import datetime, timedelta
from hashlib import sha256
from itertools import groupby
from math import cos, pi, sin
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()

@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """Get simulated current weather and 24 hourly records per day starting today.

    Args:
        location: Location to simulate, such as "San Francisco, CA".
        days: Number of calendar days to forecast (1–7, defaults to 3).

    Returns:
        Dated records with temperature in °C, sky condition, humidity in percent,
        wind in km/h, and solar irradiance in W/m², plus the peak irradiance and
        its hour for each date. Dates follow the runtime clock.
    """
    location = location.strip()
    if not location:
        return {"error": "Provide a non-empty location."}
    if not 1 <= days <= 7:
        return {"error": "Forecast days must be between 1 and 7."}

    now = datetime.now()
    today = now.date()
    sky_conditions = (("clear", 1.0), ("partly_cloudy", 0.65), ("cloudy", 0.30))
    hourly = []
    daily_solar_peaks = []

    for day_offset in range(days):
        forecast_date = (today + timedelta(days=day_offset)).isoformat()
        rng = random.Random(f"{location.casefold()}:{forecast_date}")
        average_temperature = rng.uniform(16, 25)
        sky_by_block = rng.choices(sky_conditions, k=4)
        day_forecast = []

        for hour in range(24):
            condition, cloud_factor = sky_by_block[hour // 6]
            daylight = max(0.0, sin(pi * (hour - 6) / 12))
            temperature = average_temperature + 5 * cos(pi * (hour - 15) / 12)
            day_forecast.append({
                "date": forecast_date,
                "hour": hour,
                "temperature_c": round(temperature, 1),
                "condition": condition,
                "solar_irradiance": round(850 * daylight * cloud_factor),
                "humidity": rng.randint(35, 85),
                "wind_speed": round(rng.uniform(2, 24), 1),
            })

        peak = max(day_forecast, key=lambda row: row["solar_irradiance"])
        daily_solar_peaks.append({
            "date": forecast_date,
            "hour": peak["hour"],
            "solar_irradiance": peak["solar_irradiance"],
        })
        hourly.extend(day_forecast)

    return {
        "source": "simulated",
        "location": location,
        "forecast_days": days,
        "solar_irradiance_unit": "W/m²",
        "wind_speed_unit": "km/h",
        "current": hourly[now.hour].copy(),
        "hourly": hourly,
        "daily_solar_peaks": daily_solar_peaks,
    }


@tool
def get_electricity_prices(date: str | None = None) -> Dict[str, Any]:
    """Get 24 simulated time-of-use electricity prices in USD per kWh.

    Args:
        date: Date in YYYY-MM-DD format (defaults to today).

    Returns:
        Hourly rates of $0.28/kWh from 06:00 up to 22:00 and $0.12/kWh
        otherwise, plus rate windows with exclusive end times. A window that
        includes the 23:00 hour ends at 24:00. The demand charge is zero.
    """
    if date is None:
        date = datetime.now().date().isoformat()
    try:
        valid_date = datetime.strptime(date, "%Y-%m-%d").date().isoformat() == date
    except ValueError:
        valid_date = False
    if not valid_date:
        return {"error": "Provide a valid date in YYYY-MM-DD format."}

    hourly_rates = []
    for hour in range(24):
        is_peak = 6 <= hour < 22
        hourly_rates.append({
            "hour": hour,
            "rate": 0.28 if is_peak else 0.12,
            "period": "peak" if is_peak else "off_peak",
            "demand_charge": 0.0,
        })

    rate_windows = []
    grouped_rates = groupby(hourly_rates, key=lambda row: (row["period"], row["rate"]))
    for (period, rate), entries in grouped_rates:
        hours = [entry["hour"] for entry in entries]
        rate_windows.append({
            "period": period,
            "start_time": f"{hours[0]:02d}:00",
            "end_time_exclusive": f"{hours[-1] + 1:02d}:00",
            "rate": rate,
        })

    return {
        "source": "simulated",
        "date": date,
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "hourly_rates": hourly_rates,
        "rate_windows": rate_windows,
    }


@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): Inclusive calendar end date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_usage_by_date_range(start_dt, end_dt, end_inclusive=False)
        
        if device_type:
            records = [r for r in records if r.device_type == device_type]
        
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }
        
        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })
        
        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): Inclusive calendar end date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Period totals, daily totals for dates with records,
        highest-generation dates, and individual production records.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_generation_by_date_range(start_dt, end_dt, end_inclusive=False)
        total_generation = sum(record.generation_kwh for record in records)
        daily_totals = {}
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(total_generation, 2),
            "average_daily_generation": round(total_generation / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }
        
        for record in records:
            day = record.timestamp.date().isoformat()
            daily = daily_totals.setdefault(day, {
                "date": day, "total_generation_kwh": 0.0, "total_records": 0,
            })
            daily["total_generation_kwh"] += record.generation_kwh
            daily["total_records"] += 1
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })

        daily_generation = [daily_totals[day] for day in sorted(daily_totals)]
        for daily in daily_generation:
            daily["total_generation_kwh"] = round(daily["total_generation_kwh"], 2)
        highest_total = max((daily["total_generation_kwh"] for daily in daily_generation), default=None)
        generation_data["daily_generation"] = daily_generation
        generation_data["highest_generation_dates"] = [
            daily["date"] for daily in daily_generation if daily["total_generation_kwh"] == highest_total
        ]

        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }
        
        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1
        
        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)
        
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}


ENERGY_TIPS_COLLECTION = "ecohome_energy_tips"
ENERGY_TIPS_INDEX_DIR = "data/vectorstore"


def load_energy_tip_documents() -> list[Document]:
    """Load the plain-text energy guides with their source paths."""
    documents = []
    for path in sorted(Path("data/documents").glob("*.txt")):
        documents.extend(TextLoader(str(path), encoding="utf-8").load())
    if not documents:
        raise ValueError("No energy guides found in data/documents.")
    return documents


def index_energy_tip_chunks(vectorstore: Chroma, chunks: list[Document]) -> None:
    """Reuse unchanged embeddings and replace chunks from edited guides."""
    if not chunks:
        raise ValueError("No energy guide text to index.")

    chunks_by_id = {}
    for chunk in chunks:
        key = json.dumps([chunk.metadata["source"], chunk.page_content])
        chunks_by_id[sha256(key.encode("utf-8")).hexdigest()] = chunk

    indexed_ids = set(vectorstore.get(include=[])["ids"])
    new_ids = [chunk_id for chunk_id in chunks_by_id if chunk_id not in indexed_ids]
    if new_ids:
        new_chunks = [chunks_by_id[chunk_id] for chunk_id in new_ids]
        vectorstore.add_documents(new_chunks, ids=new_ids)

    obsolete_ids = sorted(indexed_ids.difference(chunks_by_id))
    if obsolete_ids:
        vectorstore.delete(ids=obsolete_ids)


@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        load_dotenv()
        api_key = os.getenv("VOCAREUM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set VOCAREUM_API_KEY to use the Vocareum embeddings service.")

        embeddings = OpenAIEmbeddings(
            base_url="https://openai.vocareum.com/v1",
            api_key=api_key,
        )

        vectorstore = Chroma(
            collection_name=ENERGY_TIPS_COLLECTION,
            persist_directory=ENERGY_TIPS_INDEX_DIR,
            embedding_function=embeddings,
        )
        if not vectorstore.get(limit=1, include=[])["ids"]:
            documents = load_energy_tip_documents()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            index_energy_tip_chunks(vectorstore, chunks)
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)
        
        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }
        
        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })
        
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float = 0.12,
    investment_cost_usd: float | None = None,
    analysis_years: float = 1.0,
    emissions_factor_kg_co2e_per_kwh: float | None = None,
) -> Dict[str, Any]:
    """Estimate daily and annual savings, with optional investment and emissions results.

    Args:
        device_type: Type of device being optimized.
        current_usage_kwh: Current daily energy consumption in kWh.
        optimized_usage_kwh: Optimized daily energy consumption in kWh.
        price_per_kwh: Electricity price in USD per kWh.
        investment_cost_usd: Upfront cost for simple payback and ROI, if supplied.
        analysis_years: ROI horizon in years, one year by default.
        emissions_factor_kg_co2e_per_kwh: Grid emissions factor to use, if supplied.

    Returns:
        Savings estimates assuming the daily reduction repeats for 365 days per year.
    """
    if analysis_years <= 0 or (investment_cost_usd is not None and investment_cost_usd <= 0):
        return {"error": "The investment cost and analysis horizon must be positive."}
    if emissions_factor_kg_co2e_per_kwh is not None and emissions_factor_kg_co2e_per_kwh < 0:
        return {"error": "The emissions factor must be non-negative."}

    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    annual_savings_usd = savings_usd * 365
    result = {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(annual_savings_usd, 2),
    }
    if investment_cost_usd is not None:
        net_savings_usd = annual_savings_usd * analysis_years - investment_cost_usd
        payback = investment_cost_usd / annual_savings_usd if annual_savings_usd > 0 else None
        result["investment"] = {
            "cost_usd": investment_cost_usd,
            "analysis_years": analysis_years,
            "roi_percentage": round(100 * net_savings_usd / investment_cost_usd, 2),
            "net_savings_usd": round(net_savings_usd, 2),
            "simple_payback_years": round(payback, 2) if payback is not None else None,
        }
    if emissions_factor_kg_co2e_per_kwh is not None:
        avoided = savings_kwh * emissions_factor_kg_co2e_per_kwh
        result["emissions"] = {
            "factor_kg_co2e_per_kwh": emissions_factor_kg_co2e_per_kwh,
            "daily_avoided_kg_co2e": round(avoided, 2),
            "annual_avoided_kg_co2e": round(avoided * 365, 2),
        }
    return result


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]

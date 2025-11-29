"""
API endpoint for testing and monitoring data sources
"""

from fastapi import APIRouter, HTTPException
from src.ingestion.multi_source_fetcher import MultiSourceFetcher
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/sources/test")
async def test_data_sources(symbol: str = "RELIANCE"):
    """
    Test all available data sources
    
    Args:
        symbol: Symbol to test with (default: RELIANCE)
    
    Returns:
        Test results for all sources
    """
    try:
        logger.info(f"Testing all data sources with symbol: {symbol}")
        
        multi_fetcher = MultiSourceFetcher()
        results = multi_fetcher.test_all_sources(symbol)
        
        # Count successes and failures
        success_count = sum(1 for r in results.values() if r['status'] == 'success')
        failed_count = len(results) - success_count
        
        return {
            "status": "complete",
            "test_symbol": symbol,
            "total_sources": len(results),
            "successful": success_count,
            "failed": failed_count,
            "results": results,
            "recommendation": _get_recommendation(results)
        }
        
    except Exception as e:
        logger.error(f"Error testing data sources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/list")
async def list_data_sources():
    """
    List all available data sources
    
    Returns:
        List of configured data sources
    """
    try:
        multi_fetcher = MultiSourceFetcher()
        sources = multi_fetcher.get_available_sources()
        
        return {
            "status": "success",
            "sources": sources,
            "total": len(sources),
            "failsafe": "Automatic fallback to CSV if all sources fail"
        }
        
    except Exception as e:
        logger.error(f"Error listing sources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_recommendation(results: dict) -> str:
    """Generate recommendation based on test results"""
    working_sources = [name for name, result in results.items() 
                      if result['status'] == 'success']
    
    if not working_sources:
        return "⚠️ All live sources failed. System will use CSV data. Consider checking API keys and network connectivity."
    elif len(working_sources) == len(results):
        return "✅ All data sources operational. System has maximum redundancy."
    else:
        return f"⚠️ {len(working_sources)} of {len(results)} sources working. System operational with reduced redundancy."

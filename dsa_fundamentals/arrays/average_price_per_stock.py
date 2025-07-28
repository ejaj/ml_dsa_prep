from collections import defaultdict
"""
Input
 trades = [
        {"symbol": "AAPL", "price": 150},
        {"symbol": "AAPL", "price": 155},
        {"symbol": "GOOG", "price": 1000},
        {"symbol": "GOOG", "price": 1100},
        {"symbol": "TSLA", "price": 700},
        {"symbol": None, "price": 300},  # Invalid
        {"symbol": "AAPL", "price": "bad"}  # Invalid
    ]

Output: 
expected = {
        "AAPL": (150 + 155) / 2,
        "GOOG": (1000 + 1100) / 2,
        "TSLA": 700.0
    }
"""

def average_price_per_stock(trades):
    """
    Calculate average price per stock symbol.

    Args:
        trades (List(Dict)): Each dict has 'symbol' and 'price'
    Retruns:
        Dict[str, flot]: Symbol to average price mapping
    """
    if not trades:
        return {}
    price_sum = defaultdict(float)
    count = defaultdict(int)

    for trade in trades:
        symbol = trade.get('symbol')
        price = trade.get('price')

        if not symbol or not isinstance(price, (int, float)):
            continue
        price_sum[symbol] += price
        count[symbol] += 1
    average_price = {sym:price_sum[sym]/count[sym] for sym in price_sum}

    return average_price

def test_average_price_per_stock():
    trades = [
        {"symbol": "AAPL", "price": 150},
        {"symbol": "AAPL", "price": 155},
        {"symbol": "GOOG", "price": 1000},
        {"symbol": "GOOG", "price": 1100},
        {"symbol": "TSLA", "price": 700},
        {"symbol": None, "price": 300},  # Invalid
        {"symbol": "AAPL", "price": "bad"}  # Invalid
    ]
    
    expected = {
        "AAPL": (150 + 155) / 2,
        "GOOG": (1000 + 1100) / 2,
        "TSLA": 700.0
    }

    result = average_price_per_stock(trades)
    assert result == expected, f"Expected {expected}, got {result}"
    print("✅ Test passed.")

test_average_price_per_stock()















from typing import List


def buy_and_sell_stock_once(prices: List[int]) -> int:
    """
    :type prices: List[int]
    :rtype: int
    """
    min_price_so_far = float("inf")
    max_profit = 0.0
    for price in prices:
        min_price_so_far = min(min_price_so_far, price)
        max_profit = max(max_profit, price - min_price_so_far)
    return max_profit


prices = [310, 315, 275, 295, 260, 270, 290, 230, 255, 250]

buy_and_sell_stock_once(prices)
assert buy_and_sell_stock_once(prices) == 30

prices = [100, 180, 260, 310, 40, 535, 695]
assert buy_and_sell_stock_once(prices) == 655

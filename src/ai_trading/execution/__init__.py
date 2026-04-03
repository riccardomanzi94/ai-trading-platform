"""Execution module for paper trading."""

from .paper_execution import (
    Execution,
    execute_order,
    execute_all_orders,
    get_executions,
    update_portfolio,
    initialize_portfolio,
    get_portfolio_summary,
)

__all__ = [
    "Execution",
    "execute_order",
    "execute_all_orders",
    "get_executions",
    "update_portfolio",
    "initialize_portfolio",
    "get_portfolio_summary",
]

#!/usr/bin/env python3
"""Enhanced Pipeline using all platform features.

Pipeline steps:
1. Data ingestion from Alpaca Markets API
2. Technical feature computation (EMA, RSI, ATR, Volatility)
3. Multi-strategy signal generation (EMA Crossover, RSI Mean Reversion, Momentum Breakout)
4. ML-enhanced predictions (RandomForest, GradientBoosting) - optional
5. Risk management (position sizing, VaR, drawdown limits)
6. Paper execution with slippage simulation
7. Alert notifications (Telegram, Slack, Email) - optional
8. Live broker integration (Alpaca) - optional

Requires:
- ALPACA_API_KEY and ALPACA_API_SECRET environment variables
- TimescaleDB running (docker-compose up -d)
- Optional: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID for alerts
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from ai_trading.shared.config import config
from ai_trading.data_ingestion import ingest_all_tickers
from ai_trading.feature_store import build_all_features
from ai_trading.signals import (
    generate_combined_signals_all,
    MultiStrategyConfig,
    CombineMethod,
)
from ai_trading.risk_engine import apply_risk_to_all_signals
from ai_trading.execution import execute_all_orders, initialize_portfolio, get_portfolio_summary
from ai_trading.alerts import send_alert, send_daily_summary, send_signal_alert, AlertLevel

logger = logging.getLogger(__name__)


def run_enhanced_pipeline(
    tickers: Optional[list[str]] = None,
    use_ml: bool = False,
    send_alerts: bool = True,
    use_broker: bool = False,
) -> dict:
    """Run enhanced pipeline with all new features.
    
    Args:
        tickers: List of tickers to process
        use_ml: Whether to use ML predictions
        send_alerts: Whether to send notifications
        use_broker: Whether to use live/paper broker
        
    Returns:
        Pipeline execution summary
    """
    print("=" * 60)
    print("AI TRADING PLATFORM - Enhanced Pipeline")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 1. Data Ingestion
    print("\n[1/6] Ingesting market data...")
    ingestion = ingest_all_tickers(tickers, start_date="2024-01-01")
    total_rows = sum(ingestion.values())
    print(f"✓ Downloaded {total_rows} rows for {len(ingestion)} tickers")
    
    # 2. Feature Engineering
    print("\n[2/6] Computing technical features...")
    features = build_all_features(tickers)
    print("✓ Calculated EMA, RSI, ATR, Volatility")
    
    # 3. Multi-Strategy Signal Generation
    print("\n[3/6] Generating signals (multi-strategy)...")
    strategy_config = MultiStrategyConfig(
        combine_method=CombineMethod.WEIGHTED,
        min_strength=0.4,
    )
    signals = generate_combined_signals_all(tickers, strategy_config)
    
    signal_count = sum(len(s) for s in signals.values())
    print(f"✓ Generated {signal_count} signals from multi-strategy combiner")
    
    for ticker, sigs in signals.items():
        for s in sigs:
            print(f"   📊 {ticker}: {s.signal_type.value} @ ${s.price_at_signal:.2f} "
                  f"(strength: {s.strength:.2f})")
            if send_alerts:
                send_signal_alert(s)
    
    # 4. ML Enhancement (optional)
    ml_predictions = {}
    if use_ml:
        print("\n[4/6] Running ML predictions...")
        try:
            from ai_trading.ml import (
                prepare_ml_features,
                create_target_variable,
                train_model,
                predict_direction,
            )
            from ai_trading.ml.features import load_full_dataset
            
            for ticker in (tickers or config.trading.tickers):
                try:
                    df = load_full_dataset(ticker)
                    ml_features = prepare_ml_features(df)
                    target = create_target_variable(df)
                    
                    # Quick train on recent data
                    X = ml_features.dropna()
                    y = target.loc[X.index].dropna()
                    common_idx = X.index.intersection(y.index)
                    X = X.loc[common_idx]
                    y = y.loc[common_idx]
                    
                    if len(X) > 100:
                        train_size = int(len(X) * 0.8)
                        model = train_model(
                            X.iloc[:train_size], 
                            y.iloc[:train_size], 
                            ticker
                        )
                        direction, confidence = predict_direction(model, X)
                        ml_predictions[ticker] = {
                            "direction": "UP" if direction == 1 else "DOWN",
                            "confidence": confidence,
                        }
                        print(f"   🤖 {ticker}: {ml_predictions[ticker]['direction']} "
                              f"({confidence:.1%} confidence)")
                except Exception as e:
                    logger.warning(f"ML failed for {ticker}: {e}")
                    
        except ImportError:
            print("   ⚠ ML dependencies not installed (pip install scikit-learn)")
    else:
        print("\n[4/6] ML predictions... (skipped)")
    
    # 5. Risk Management
    print("\n[5/6] Applying risk management...")
    orders = apply_risk_to_all_signals(signals)
    approved = [o for o in orders if o.approved]
    rejected = [o for o in orders if not o.approved]
    print(f"✓ {len(approved)} orders approved, {len(rejected)} rejected")
    
    for o in rejected:
        print(f"   ❌ {o.ticker}: {o.rejection_reason}")
    
    # 6. Execution
    print("\n[6/6] Executing orders...")
    
    if use_broker:
        # Use Alpaca broker
        try:
            from ai_trading.broker import AlpacaBroker
            broker = AlpacaBroker()
            
            if broker.config.api_key:
                for o in approved:
                    # Convert to signal and execute
                    from ai_trading.signals import Signal, SignalType
                    signal = Signal(
                        time=o.time,
                        ticker=o.ticker,
                        signal_type=SignalType(o.signal_type),
                        strength=1 - o.risk_score,
                        price_at_signal=0,  # Broker will get latest price
                    )
                    order = broker.execute_signal(signal, quantity=o.adjusted_quantity)
                    if order:
                        print(f"   ✓ {order.side.value.upper()} {order.quantity} {order.ticker}")
            else:
                print("   ⚠ Broker not configured, using paper execution")
                executions = execute_all_orders(orders)
        except ImportError:
            print("   ⚠ Broker module not available")
            executions = execute_all_orders(orders)
    else:
        executions = execute_all_orders(orders)
        print(f"✓ {len(executions)} paper trades executed")
        
        for e in executions:
            print(f"   💰 {e.side} {e.quantity} {e.ticker} @ ${e.price:.2f}")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    # Get portfolio state
    portfolio_df = get_portfolio_summary()
    portfolio_value = portfolio_df["current_value"].sum() if not portfolio_df.empty else 100000
    
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Tickers processed: {len(ingestion)}")
    print(f"Signals generated: {signal_count}")
    print(f"Orders approved: {len(approved)}")
    print(f"Trades executed: {len(executions) if not use_broker else 'via broker'}")
    print(f"Portfolio value: ${portfolio_value:,.2f}")
    
    # Send daily summary
    if send_alerts:
        positions = {
            row["ticker"]: {"quantity": row["quantity"]}
            for _, row in portfolio_df.iterrows()
        }
        send_daily_summary(
            portfolio_value=portfolio_value,
            daily_pnl=0,  # Would calculate from previous day
            positions=positions,
            signals_today=signal_count,
            trades_today=len(executions) if not use_broker else len(approved),
        )
    
    return {
        "status": "completed",
        "duration_seconds": duration,
        "tickers": len(ingestion),
        "signals": signal_count,
        "orders_approved": len(approved),
        "ml_predictions": ml_predictions,
    }


def run_ml_training_pipeline(ticker: str = "SPY") -> dict:
    """Run ML model training pipeline.
    
    Args:
        ticker: Ticker to train model for
        
    Returns:
        Training results
    """
    print(f"\n{'=' * 60}")
    print(f"ML Training Pipeline - {ticker}")
    print("=" * 60)
    
    try:
        from ai_trading.ml import (
            prepare_ml_features,
            create_target_variable,
            split_train_test,
            train_model,
            evaluate_model,
        )
        from ai_trading.ml.features import load_full_dataset, get_feature_importance
        
        # Load data
        print("\n[1/4] Loading data...")
        df = load_full_dataset(ticker)
        print(f"✓ Loaded {len(df)} rows")
        
        # Prepare features
        print("\n[2/4] Preparing features...")
        features = prepare_ml_features(df)
        target = create_target_variable(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(features, target)
        print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train models
        print("\n[3/4] Training models...")
        models = {}
        
        for model_type in ["random_forest", "gradient_boosting"]:
            print(f"   Training {model_type}...")
            model = train_model(X_train, y_train, ticker, model_type=model_type)
            metrics = evaluate_model(model, X_test, y_test)
            models[model_type] = {
                "model": model,
                "metrics": metrics,
            }
            print(f"   ✓ {model_type}: accuracy={metrics.accuracy:.3f}, f1={metrics.f1:.3f}")
        
        # Best model
        print("\n[4/4] Results...")
        best_type = max(models, key=lambda x: models[x]["metrics"].f1)
        best = models[best_type]
        
        print(f"\n🏆 Best Model: {best_type}")
        print(f"   Accuracy: {best['metrics'].accuracy:.3f}")
        print(f"   Precision: {best['metrics'].precision:.3f}")
        print(f"   Recall: {best['metrics'].recall:.3f}")
        print(f"   F1: {best['metrics'].f1:.3f}")
        if best['metrics'].auc_roc:
            print(f"   AUC-ROC: {best['metrics'].auc_roc:.3f}")
        
        # Feature importance
        importance = get_feature_importance(best['model'].model, best['model'].feature_names)
        print(f"\n📊 Top 5 Features:")
        for _, row in importance.head(5).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.4f}")
        
        return {
            "ticker": ticker,
            "best_model": best_type,
            "metrics": {
                "accuracy": best['metrics'].accuracy,
                "precision": best['metrics'].precision,
                "recall": best['metrics'].recall,
                "f1": best['metrics'].f1,
            },
        }
        
    except ImportError as e:
        print(f"❌ ML dependencies not installed: {e}")
        print("Run: pip install scikit-learn")
        return {"error": str(e)}


def run_strategy_optimization(ticker: str = "SPY", n_trials: int = 30) -> dict:
    """Run strategy parameter optimization.
    
    Args:
        ticker: Ticker to optimize for
        n_trials: Number of optimization trials
        
    Returns:
        Optimization results
    """
    print(f"\n{'=' * 60}")
    print(f"Strategy Optimization - {ticker}")
    print("=" * 60)
    
    try:
        from ai_trading.ml import optimize_strategy_params
        
        print(f"\nOptimizing EMA strategy ({n_trials} trials)...")
        result = optimize_strategy_params(ticker, strategy="ema", n_trials=n_trials)
        
        print(f"\n🎯 Best Parameters:")
        for key, value in result.best_params.items():
            print(f"   • {key}: {value}")
        print(f"\n📈 Best Sharpe Ratio: {result.best_value:.3f}")
        
        return {
            "ticker": ticker,
            "best_params": result.best_params,
            "best_sharpe": result.best_value,
        }
        
    except ImportError as e:
        print(f"❌ Optuna not installed: {e}")
        print("Run: pip install optuna")
        return {"error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "ml":
            ticker = sys.argv[2] if len(sys.argv) > 2 else "SPY"
            run_ml_training_pipeline(ticker)
        elif command == "optimize":
            ticker = sys.argv[2] if len(sys.argv) > 2 else "SPY"
            run_strategy_optimization(ticker)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python run_enhanced_pipeline.py [ml|optimize] [ticker]")
    else:
        # Default: run enhanced pipeline
        run_enhanced_pipeline(
            use_ml=True,
            send_alerts=True,
            use_broker=False,
        )

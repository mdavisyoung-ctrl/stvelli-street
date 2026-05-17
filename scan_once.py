"""
One-shot scan: fetches data, runs all signals, prints a report.
Works in any environment (no live dashboard, no TTY required).

Usage:  python scan_once.py
        python scan_once.py --quick     # SPY/BTC/XRP only, skips 100-stock scan
"""
import sys
import argparse
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import TOP_100_STOCKS
from scanner.data_fetcher import batch_fetch
from scanner.signals import analyze_ticker
from scanner.macro_scanner import fetch_macro_data, analyze_macro
from scanner.ml_pattern import predict, train_model
from scanner.sentiment import aggregate_sentiment
from strategies.fade_strategy import evaluate_fade
from strategies.momentum_strategy import evaluate_momentum
from portfolio.tracker import load_state, portfolio_value, win_rate
from portfolio.paper_trader import load_paper_state, paper_portfolio_value, model_accuracy_report
from config import STARTING_CAPITAL

console = Console()


def run(quick: bool = False):
    state = load_state()
    paper = load_paper_state()

    # ── Portfolio summary ──
    console.print(Panel.fit(
        f"[bold cyan]Stvelli Street[/bold cyan]  |  Cash: ${state['cash']:.2f}  "
        f"|  Win rate: {win_rate(state):.1f}%  ({state['winning_trades']}/{state['total_trades']})",
        border_style="cyan",
    ))

    # ── Macro: SPY, BTC, XRP ──
    console.print("\n[bold yellow]Scanning SPY · BTC · XRP...[/bold yellow]")
    macro_data = fetch_macro_data()
    macro_results = {}
    ml_results = {}
    spy_regime = "UNKNOWN"

    for name in ("SPY", "BTC", "XRP"):
        if name not in macro_data:
            console.print(f"  [dim]{name}: no data[/dim]")
            continue
        hist = macro_data[name].get("history")
        news = macro_data[name].get("news", [])
        sent_score = aggregate_sentiment(news)["score"]
        result = analyze_macro(name, macro_data[name], [])
        ml_result = predict(name, hist, current_sentiment=sent_score,
                            current_cp=result.get("cp_ratio") or 1.0) \
            if hist is not None and not hist.empty else {}
        macro_results[name] = result
        ml_results[name] = ml_result

    spy_regime = macro_results.get("SPY", {}).get("regime", "UNKNOWN")

    macro_table = Table(box=box.SIMPLE_HEAVY, expand=False)
    macro_table.add_column("Asset", style="bold magenta")
    macro_table.add_column("Price")
    macro_table.add_column("Regime")
    macro_table.add_column("RSI")
    macro_table.add_column("Sentiment")
    macro_table.add_column("ML Signal")
    macro_table.add_column("ML Conf")
    macro_table.add_column("Support")
    macro_table.add_column("Resistance")

    regime_colors = {
        "RISK_ON": "green", "BULLISH": "green", "OVERSOLD": "cyan",
        "RISK_OFF": "red",  "OVERBOUGHT": "yellow", "NEUTRAL": "white",
    }
    for name in ("SPY", "BTC", "XRP"):
        m = macro_results.get(name, {})
        ml = ml_results.get(name, {})
        if not m:
            continue
        regime = m.get("regime", "NEUTRAL")
        rc = regime_colors.get(regime, "white")
        ml_label = ml.get("label", "—")
        ml_conf = f"{ml.get('confidence', 0):.0%}" if ml else "—"
        ml_color = "green" if ml_label == "UP" else ("red" if ml_label == "DOWN" else "dim")
        macro_table.add_row(
            name,
            f"${m['price']:.4f}" if m.get("price") else "—",
            f"[{rc}]{regime}[/{rc}]",
            str(m.get("rsi") or "—"),
            f"[{'green' if m.get('sentiment_label')=='BULLISH' else 'red' if m.get('sentiment_label')=='BEARISH' else 'white'}]"
            f"{m.get('sentiment_label', '—')}[/]",
            f"[{ml_color}]{ml_label}[/{ml_color}]",
            ml_conf,
            f"${m['support']:.2f}" if m.get("support") else "—",
            f"${m['resistance']:.2f}" if m.get("resistance") else "—",
        )
    console.print(macro_table)

    # SPY regime context
    regime_advice = {
        "RISK_ON":  "[green]RISK ON — favor LONG setups, momentum trades.[/green]",
        "RISK_OFF": "[red]RISK OFF — favor FADE setups, avoid catching falling knives.[/red]",
        "NEUTRAL":  "[white]NEUTRAL regime — be selective, require higher confidence.[/white]",
    }
    console.print(f"SPY Regime: {regime_advice.get(spy_regime, spy_regime)}\n")

    if quick:
        console.print("[dim]--quick mode: skipping equity scan.[/dim]")
        return

    # ── Equity scan ──
    console.print(f"[bold yellow]Scanning top 100 stocks (this takes ~3 minutes)...[/bold yellow]")
    data = batch_fetch(TOP_100_STOCKS)

    signals = []
    for ticker in TOP_100_STOCKS:
        td = data.get(ticker, {})
        if not td:
            continue
        sig = analyze_ticker(ticker, td, [])
        signals.append(sig)

    order = {"FADE": 0, "LONG": 1, "PASS": 2}
    signals.sort(key=lambda s: (order.get(s.signal_type, 9), -s.confidence))

    # ── Signals table ──
    actionable = [s for s in signals if s.signal_type != "PASS"]
    console.print(f"\n[bold]Actionable signals: {len(actionable)}[/bold]  "
                  f"[dim](showing top 20)[/dim]\n")

    sig_table = Table(box=box.SIMPLE_HEAVY, expand=True)
    sig_table.add_column("Ticker", style="bold cyan", width=7)
    sig_table.add_column("Signal", width=6)
    sig_table.add_column("Conf", width=5)
    sig_table.add_column("Price", width=9)
    sig_table.add_column("C/P", width=6)
    sig_table.add_column("Sentiment", width=9)
    sig_table.add_column("RSI", width=5)
    sig_table.add_column("Stop", width=9)
    sig_table.add_column("Target", width=9)
    sig_table.add_column("Reason / Headline", overflow="fold")

    for s in signals[:20]:
        color = {"LONG": "green", "FADE": "red", "PASS": "dim"}.get(s.signal_type, "white")
        setup = evaluate_fade(s) or evaluate_momentum(s)
        stop_str = f"${setup.stop_loss:.2f}" if setup else "—"
        target_str = f"${setup.take_profit:.2f}" if setup else "—"
        sc = "green" if s.sentiment_label == "BULLISH" else ("red" if s.sentiment_label == "BEARISH" else "white")
        detail = s.top_headline[:50] if s.top_headline else s.reason
        sig_table.add_row(
            s.ticker,
            f"[{color}]{s.signal_type}[/{color}]",
            f"{s.confidence:.0%}",
            f"${s.price:.2f}" if s.price else "—",
            f"{s.cp_ratio:.2f}" if s.cp_ratio else "—",
            f"[{sc}]{s.sentiment_label}[/]",
            str(s.rsi) if s.rsi else "—",
            stop_str,
            target_str,
            detail,
        )
    console.print(sig_table)

    # ── Best trade suggestion ──
    if actionable:
        best = actionable[0]
        setup = evaluate_fade(best) or evaluate_momentum(best)
        direction = "SHORT (put or inverse ETF)" if best.signal_type == "FADE" else "LONG (buy)"
        console.print(Panel(
            f"[bold cyan]{best.ticker}[/bold cyan]  →  [bold]{direction}[/bold] @ ~${best.price:.2f}\n"
            f"  Confidence: {best.confidence:.0%}\n"
            f"  Stop loss:   [red]${setup.stop_loss:.2f}[/red]\n"
            f"  Take profit: [green]${setup.take_profit:.2f}[/green]\n"
            f"  Reason: {best.reason}\n"
            f"  Headline: {best.top_headline[:80] if best.top_headline else '—'}\n\n"
            f"  [dim]Execute in Robinhood, then record:[/dim]\n"
            f"  [bold]enter {best.ticker} <your_fill_price>[/bold]",
            title="[bold yellow]Best Trade Right Now[/bold yellow]",
            border_style="yellow",
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="SPY/BTC/XRP only, skip equity scan")
    args = parser.parse_args()
    run(quick=args.quick)

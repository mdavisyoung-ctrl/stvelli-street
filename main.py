"""
Stvelli Street — Stock Arbitrage Scanner
Run:  python main.py

Commands (type while running):
  enter <TICKER> <PRICE>   — record your entry after placing the trade in Robinhood
  close <TICKER> <PRICE>   — record your exit price to close a position
  status                   — print full portfolio summary
  train                    — retrain ML models on latest data
  quit / exit              — stop the scanner
"""
import sys
import time
import logging
import threading
from datetime import datetime
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich import box

from config import TOP_100_STOCKS, SCAN_INTERVAL_SECONDS, STARTING_CAPITAL
from scanner.data_fetcher import batch_fetch, get_current_price
from scanner.signals import analyze_ticker
from scanner.macro_scanner import fetch_macro_data, analyze_macro, MACRO_TICKERS
from scanner.ml_pattern import predict, train_model
from scanner.sentiment import aggregate_sentiment
from strategies.fade_strategy import evaluate_fade
from strategies.momentum_strategy import evaluate_momentum
from portfolio.tracker import (
    load_state, open_position, close_position,
    portfolio_value, win_rate
)
from portfolio.paper_trader import (
    load_paper_state, paper_open, paper_close,
    paper_portfolio_value, model_accuracy_report, should_paper_enter
)
from portfolio.risk_manager import (
    stop_loss_price, take_profit_price, should_stop, should_take_profit
)

logging.basicConfig(
    filename="scanner.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

console = Console()

_lock = threading.Lock()
_portfolio = load_state()
_paper = load_paper_state()
_cp_history: dict = defaultdict(list)
_macro_cp_history: dict = defaultdict(list)
_active_signals: list = []
_macro_results: dict = {}
_ml_results: dict = {}
_last_scan: str = "Never"
_scan_count: int = 0
_spy_regime: str = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def _sig_color(t: str) -> str:
    return {"LONG": "green", "FADE": "red", "PASS": "dim"}.get(t, "white")


def _regime_color(r: str) -> str:
    return {
        "RISK_ON": "green", "BULLISH": "green", "OVERSOLD": "cyan",
        "RISK_OFF": "red",  "OVERBOUGHT": "yellow",
        "NEUTRAL": "white", "UNKNOWN": "dim",
    }.get(r, "white")


def build_macro_panel(macro: dict, ml: dict, paper: dict) -> Panel:
    lines = []
    paper_prices = {t: macro[t]["price"] for t in macro if macro[t].get("price")}
    paper_val = paper_portfolio_value(paper, paper_prices)
    paper_pnl = paper_val - 50.0
    acc = model_accuracy_report(paper)

    lines.append(f"[bold]Paper portfolio:[/bold] ${paper_val:.2f}  "
                 f"[{'green' if paper_pnl>=0 else 'red'}]({paper_pnl:+.2f})[/]")
    for ticker in ("SPY", "BTC", "XRP"):
        if ticker not in macro:
            continue
        m = macro[ticker]
        ml_r = ml.get(ticker, {})
        regime = m.get("regime", "NEUTRAL")
        rc = _regime_color(regime)
        ml_label = ml_r.get("label", "?")
        ml_conf = ml_r.get("confidence", 0)
        ml_color = "green" if ml_label == "UP" else ("red" if ml_label == "DOWN" else "dim")
        ticker_acc = acc.get(ticker, None)
        acc_str = f"  [dim]acc:{ticker_acc:.0f}%[/dim]" if ticker_acc is not None else ""
        price_str = f"${m['price']:.2f}" if m.get("price") else "—"
        lines.append(
            f"  [bold]{ticker}[/bold] {price_str}  "
            f"[{rc}]{regime}[/{rc}]  "
            f"ML:[{ml_color}]{ml_label}({ml_conf:.0%})[/{ml_color}]"
            f"{acc_str}"
        )
        # Show paper positions
        if ticker in paper.get("positions", {}):
            pos = paper["positions"][ticker]
            cur = m.get("price", pos["entry_price"])
            unreal = (cur - pos["entry_price"]) * pos["shares"]
            if pos["direction"] == "FADE":
                unreal = -unreal
            c = "green" if unreal >= 0 else "red"
            lines.append(f"    [dim]paper {pos['direction']} @ ${pos['entry_price']:.4f} "
                         f"[{c}]{unreal:+.2f}[/{c}][/dim]")

    return Panel("\n".join(lines), title="SPY / BTC / XRP  +  ML", border_style="magenta")


def build_signals_table(signals: list, spy_regime: str) -> Table:
    regime_note = f"  [dim](SPY regime: {spy_regime})[/dim]" if spy_regime else ""
    t = Table(title=f"Top Signals{regime_note}", box=box.SIMPLE_HEAVY, expand=True)
    t.add_column("Ticker", style="bold cyan", width=7)
    t.add_column("Signal", width=6)
    t.add_column("Conf", width=5)
    t.add_column("Price", width=8)
    t.add_column("C/P", width=6)
    t.add_column("Sent", width=8)
    t.add_column("RSI", width=5)
    t.add_column("Target", width=8)
    t.add_column("Stop", width=8)
    t.add_column("Headline", overflow="fold")

    for s in signals[:12]:
        color = _sig_color(s.signal_type)
        setup = evaluate_fade(s) or evaluate_momentum(s)
        target = f"${setup.take_profit:.2f}" if setup else "—"
        stop = f"${setup.stop_loss:.2f}" if setup else "—"
        t.add_row(
            s.ticker,
            f"[{color}]{s.signal_type}[/{color}]",
            f"{s.confidence:.0%}",
            f"${s.price:.2f}" if s.price else "—",
            f"{s.cp_ratio:.2f}" if s.cp_ratio else "—",
            f"[{'green' if s.sentiment_label=='BULLISH' else 'red' if s.sentiment_label=='BEARISH' else 'white'}]"
            f"{s.sentiment_label}[/]",
            str(s.rsi) if s.rsi else "—",
            target, stop,
            s.top_headline[:55] if s.top_headline else "—",
        )
    return t


def build_portfolio_panel(state: dict, prices: dict) -> Panel:
    value = portfolio_value(state, prices)
    total_pnl = value - STARTING_CAPITAL
    c = "green" if total_pnl >= 0 else "red"
    lines = [
        f"[bold]Cash:[/bold]    ${state['cash']:.2f}",
        f"[bold]Value:[/bold]   ${value:.2f}  [{c}]({total_pnl:+.2f})[/{c}]",
        f"[bold]Win rate:[/bold] {win_rate(state):.1f}%  ({state['winning_trades']}/{state['total_trades']})",
        "",
    ]
    if state["positions"]:
        lines.append("[bold underline]Open Positions[/bold underline]")
        for ticker, pos in state["positions"].items():
            cur = prices.get(ticker, pos["entry_price"])
            unreal = (cur - pos["entry_price"]) * pos["shares"]
            if pos["signal_type"] == "FADE":
                unreal = -unreal
            pc = "green" if unreal >= 0 else "red"
            lines.append(
                f"  {ticker}: {pos['signal_type']} @ ${pos['entry_price']:.2f} → "
                f"${cur:.2f}  [{pc}]{unreal:+.2f}[/{pc}]"
            )
    else:
        lines.append("[dim]No open positions[/dim]")
    return Panel("\n".join(lines), title="Real Portfolio ($50)", border_style="cyan")


def build_status_panel() -> Panel:
    lines = [
        f"[bold]Last scan:[/bold] {_last_scan}",
        f"[bold]Scans:[/bold] {_scan_count}",
        f"[bold]Active signals:[/bold] {len([s for s in _active_signals if s.signal_type != 'PASS'])}",
        "",
        "[dim]enter <TICKER> <PRICE>[/dim]",
        "[dim]close <TICKER> <PRICE>[/dim]",
        "[dim]train | status | quit[/dim]",
    ]
    return Panel("\n".join(lines), title="Status", border_style="yellow")


def render_dashboard() -> Layout:
    with _lock:
        sigs = list(_active_signals)
        state = dict(_portfolio)
        macro = dict(_macro_results)
        ml = dict(_ml_results)
        paper = dict(_paper)
        regime = _spy_regime

    prices = {t: get_current_price(t) or 0 for t in state.get("positions", {})}

    layout = Layout()
    layout.split_column(
        Layout(name="top", size=14),
        Layout(name="mid", size=10),
        Layout(name="bottom"),
    )
    layout["top"].split_row(
        Layout(build_portfolio_panel(state, prices), name="portfolio"),
        Layout(build_macro_panel(macro, ml, paper), name="macro"),
        Layout(build_status_panel(), name="status"),
    )
    layout["mid"].update(Panel(
        _build_suggested_trade(sigs, regime),
        title="[bold yellow]Suggested Trade[/bold yellow]",
        border_style="yellow",
    ))
    layout["bottom"].update(build_signals_table(sigs, regime))
    return layout


def _build_suggested_trade(signals: list, spy_regime: str) -> str:
    """Picks the single best trade to suggest right now."""
    candidates = [s for s in signals if s.signal_type != "PASS"]
    if not candidates:
        return "[dim]No actionable signals this scan. Waiting...[/dim]"

    # In RISK_OFF regime, prefer FADEs; in RISK_ON, prefer LONGs
    if spy_regime == "RISK_OFF":
        fades = [s for s in candidates if s.signal_type == "FADE"]
        best = fades[0] if fades else candidates[0]
    else:
        longs = [s for s in candidates if s.signal_type == "LONG"]
        best = longs[0] if longs else candidates[0]

    setup = evaluate_fade(best) or evaluate_momentum(best)
    if not setup:
        return "[dim]Signal found but setup incomplete — waiting for next scan.[/dim]"

    direction = "SHORT (put or inverse)" if best.signal_type == "FADE" else "LONG (buy)"
    ml_note = ""
    ml_r = _ml_results.get(best.ticker, {})
    if ml_r.get("model_trained"):
        ml_note = f"\n  ML pattern: [bold]{ml_r['label']}[/bold] ({ml_r['confidence']:.0%} confidence)"

    return (
        f"[bold cyan]{best.ticker}[/bold cyan]  →  [bold]{direction}[/bold]  "
        f"@ ~${best.price:.2f}  (conf: {best.confidence:.0%})\n"
        f"  Stop loss:   [red]${setup.stop_loss:.2f}[/red]    "
        f"Take profit: [green]${setup.take_profit:.2f}[/green]\n"
        f"  Reason: {best.reason}"
        f"{ml_note}\n"
        f"  [dim]→ Type:  enter {best.ticker} <your_fill_price>[/dim]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scanner loop
# ─────────────────────────────────────────────────────────────────────────────

def _scanner_loop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            _run_scan()
        except Exception as e:
            logging.exception("Scan error: %s", e)
        stop_event.wait(SCAN_INTERVAL_SECONDS)


def _run_scan() -> None:
    global _active_signals, _last_scan, _scan_count, _macro_results, _ml_results, _spy_regime
    global _portfolio, _paper

    logging.info("Starting scan #%d", _scan_count + 1)

    # ── Macro scan (SPY, BTC, XRP) ──
    macro_data = fetch_macro_data()
    new_macro = {}
    new_ml = {}
    for name in ("SPY", "BTC", "XRP"):
        if name not in macro_data:
            continue
        hist = macro_data[name].get("history")
        sent_data = macro_data[name].get("news", [])
        sent_score = aggregate_sentiment(sent_data)["score"]
        with _lock:
            cp_hist = list(_macro_cp_history[name])
        result = analyze_macro(name, macro_data[name], cp_hist)
        ml_result = predict(
            name, hist,
            current_sentiment=sent_score,
            current_cp=result.get("cp_ratio") or 1.0
        ) if hist is not None and not hist.empty else {}
        new_macro[name] = result
        new_ml[name] = ml_result

        # Auto paper trade
        if result.get("price") and ml_result:
            enter, direction = should_paper_enter(result, ml_result)
            with _lock:
                if enter and name not in _paper["positions"]:
                    _paper = paper_open(_paper, name, result["price"], direction,
                                        ml_result.get("confidence", 0),
                                        ml_result.get("prediction", 0))
                # Check paper exits (5% stop / 10% target)
                elif name in _paper["positions"]:
                    pos = _paper["positions"][name]
                    cur = result["price"]
                    stop = stop_loss_price(pos["entry_price"], None, pos["direction"])
                    target = take_profit_price(pos["entry_price"], None, pos["direction"])
                    if should_stop(pos["entry_price"], cur, stop, pos["direction"]) or \
                       should_take_profit(pos["entry_price"], cur, target, pos["direction"]):
                        _paper = paper_close(_paper, name, cur)

    spy_regime = new_macro.get("SPY", {}).get("regime", "UNKNOWN")

    # ── Equity scan ──
    data = batch_fetch(TOP_100_STOCKS)
    signals = []
    with _lock:
        for ticker in TOP_100_STOCKS:
            td = data.get(ticker, {})
            if not td:
                continue
            sig = analyze_ticker(ticker, td, list(_cp_history[ticker]))
            if sig.cp_ratio is not None:
                _cp_history[ticker].append(sig.cp_ratio)
                if len(_cp_history[ticker]) > 30:
                    _cp_history[ticker] = _cp_history[ticker][-30:]

            # Run ML on individual equities too
            hist = td.get("history")
            if hist is not None and not hist.empty:
                ml_r = predict(ticker, hist,
                               current_sentiment=sig.sentiment_score,
                               current_cp=sig.cp_ratio or 1.0)
                new_ml[ticker] = ml_r

            signals.append(sig)

        order = {"FADE": 0, "LONG": 1, "PASS": 2}
        signals.sort(key=lambda s: (order.get(s.signal_type, 9), -s.confidence))
        _active_signals = signals
        _macro_results = new_macro
        _ml_results = new_ml
        _spy_regime = spy_regime
        _last_scan = datetime.now().strftime("%H:%M:%S")
        _scan_count += 1

    _check_real_exits(data)
    logging.info("Scan complete. Signals: %d", len(signals))


def _check_real_exits(data: dict) -> None:
    global _portfolio
    with _lock:
        for ticker, pos in list(_portfolio["positions"].items()):
            td = data.get(ticker, {})
            hist = td.get("history")
            if hist is None or hist.empty:
                continue
            cur = float(hist["Close"].iloc[-1])
            sig_type = pos["signal_type"]
            stop = stop_loss_price(pos["entry_price"], None, sig_type)
            target = take_profit_price(pos["entry_price"], None, sig_type)
            if should_stop(pos["entry_price"], cur, stop, sig_type):
                console.print(f"\n[bold red]⚠ STOP HIT: {ticker} @ ${cur:.2f} → CLOSE IN ROBINHOOD NOW[/bold red]")
                _portfolio = close_position(_portfolio, ticker, cur)
            elif should_take_profit(pos["entry_price"], cur, target, sig_type):
                console.print(f"\n[bold green]✓ TARGET HIT: {ticker} @ ${cur:.2f} → Consider closing[/bold green]")


# ─────────────────────────────────────────────────────────────────────────────
# Command handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_command(cmd: str) -> bool:
    global _portfolio, _paper
    parts = cmd.strip().split()
    if not parts:
        return True
    verb = parts[0].lower()

    if verb in ("quit", "exit", "q"):
        return False

    if verb == "enter" and len(parts) == 3:
        ticker = parts[1].upper()
        try:
            price = float(parts[2])
        except ValueError:
            console.print("[red]Invalid price[/red]")
            return True
        with _lock:
            sig = next((s for s in _active_signals if s.ticker == ticker), None)
        sig_type = sig.signal_type if sig and sig.signal_type != "PASS" else "LONG"
        with _lock:
            _portfolio = open_position(_portfolio, ticker, price, sig_type)
        setup = (evaluate_fade(sig) or evaluate_momentum(sig)) if sig else None
        console.print(f"\n[green]Opened {sig_type} {ticker} @ ${price:.2f}[/green]")
        if setup:
            console.print(f"  Stop loss:   [red]${setup.stop_loss:.2f}[/red]")
            console.print(f"  Take profit: [green]${setup.take_profit:.2f}[/green]")
        return True

    if verb == "close" and len(parts) == 3:
        ticker = parts[1].upper()
        try:
            price = float(parts[2])
        except ValueError:
            console.print("[red]Invalid price[/red]")
            return True
        with _lock:
            _portfolio = close_position(_portfolio, ticker, price)
        console.print(f"\n[cyan]Closed {ticker} @ ${price:.2f}[/cyan]")
        return True

    if verb == "train":
        console.print("[yellow]Retraining ML models on latest data...[/yellow]")
        _retrain_models()
        console.print("[green]Models retrained.[/green]")
        return True

    if verb == "status":
        with _lock:
            state = dict(_portfolio)
        prices = {t: get_current_price(t) or 0 for t in state.get("positions", {})}
        console.print(build_portfolio_panel(state, prices))
        return True

    console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
    return True


def _retrain_models() -> None:
    for name in ("SPY", "BTC", "XRP"):
        meta = MACRO_TICKERS.get(name, {})
        if not meta:
            continue
        import yfinance as yf
        t = yf.Ticker(meta["yf_symbol"])
        hist = t.history(period="2y", auto_adjust=True)
        if not hist.empty:
            train_model(name, hist)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold cyan]Stvelli Street[/bold cyan] — Arbitrage Scanner\n"
        "SPY · BTC · XRP · Top 100 stocks  |  C/P ratio + sentiment + ML\n"
        "[dim]First scan takes ~3 minutes...[/dim]",
        border_style="cyan",
    ))

    stop_event = threading.Event()
    scanner_thread = threading.Thread(target=_scanner_loop, args=(stop_event,), daemon=True)
    scanner_thread.start()

    time.sleep(3)

    try:
        with Live(console=console, refresh_per_second=0.3, screen=True) as live:
            while True:
                live.update(render_dashboard())
                import select
                ready, _, _ = select.select([sys.stdin], [], [], 1.0)
                if ready:
                    line = sys.stdin.readline().strip()
                    live.stop()
                    if not handle_command(line):
                        break
                    live.start()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        scanner_thread.join(timeout=5)
        console.print("[yellow]Stopped. State saved.[/yellow]")


if __name__ == "__main__":
    main()

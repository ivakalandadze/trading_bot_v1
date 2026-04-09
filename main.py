"""
main.py — Autonomous Trading Bot entry point.

Usage:
  python main.py                  # Interactive menu
  python main.py --mode paper     # Start paper trading immediately
  python main.py --mode live      # Start live trading (requires broker keys)
  python main.py --scan           # Run one scan and exit
  python main.py --status         # Print current status and exit
  python main.py --symbol AAPL    # Analyse a single symbol

Environment: configure in .env (copy from .env.example)
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime

import config

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log", mode="a"),
    ],
)
logger = logging.getLogger("main")

# Suppress noisy third-party loggers
for lib in ("urllib3", "httpx", "yfinance", "peewee", "binance"):
    logging.getLogger(lib).setLevel(logging.WARNING)

# ── Conditional imports ───────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = CYAN = YELLOW = RED = MAGENTA = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = DIM = ""

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    def tabulate(data, headers=(), tablefmt="simple"):
        if headers:
            print("  ".join(str(h) for h in headers))
        for row in data:
            print("  ".join(str(c) for c in row))
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

BANNER = f"""
{Fore.CYAN}{Style.BRIGHT}
╔══════════════════════════════════════════════════════════╗
║          AUTONOMOUS TRADING BOT  v1.0                    ║
║  10-technique analysis · LLM judge · Paper + Live        ║
╚══════════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""


def print_banner():
    print(BANNER)


def c(text, color):
    """Colorise text if colorama is available."""
    return f"{color}{text}{Style.RESET_ALL}" if HAS_COLOR else str(text)


def print_section(title: str):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}{Style.RESET_ALL}")


def fmt_pnl(val):
    if val is None:
        return "—"
    color = Fore.GREEN if val >= 0 else Fore.RED
    return c(f"{'+'if val>=0 else ''}{val:.2f}", color)


# ─────────────────────────────────────────────────────────────────────────────
# Status dashboard
# ─────────────────────────────────────────────────────────────────────────────

def print_status(engine):
    status = engine.get_status()
    mode   = status["mode"].upper()
    db_file = config.DATABASE_PATH

    print_section(f"BOT STATUS — {mode} MODE  [{db_file}]")

    # Account
    account = status.get("account", {})
    if account:
        print(f"  {'Total Value':20} ${account.get('total_value', 0):>12,.2f}")
        print(f"  {'Cash':20} ${account.get('cash', 0):>12,.2f}")
        print(f"  {'Equity':20} ${account.get('equity', 0):>12,.2f}")
        ret = account.get("return_pct", 0)
        color = Fore.GREEN if ret >= 0 else Fore.RED
        print(f"  {'Return':20} {c(f'{ret:+.2f}%', color):>12}")

    # Open positions
    positions = status.get("positions", [])
    if positions:
        print_section(f"OPEN POSITIONS ({len(positions)})")
        rows = []
        for pos in positions:
            ep  = pos["entry_price"]
            cp  = pos["current_price"]
            qty = pos["quantity"]
            pnl = (cp - ep) * qty
            pct = (cp / ep - 1) * 100 if ep > 0 else 0
            rows.append([
                pos["symbol"],
                pos["asset_type"],
                f"{qty:.6g}",
                f"${ep:.4f}",
                f"${cp:.4f}",
                fmt_pnl(pnl),
                f"{pct:+.2f}%",
                f"${pos.get('stop_loss', 0):.4f}",
                f"${pos.get('take_profit', 0):.4f}",
            ])
        print(tabulate(rows,
                       headers=["Symbol", "Type", "Qty", "Entry", "Current",
                                "P&L ($)", "P&L (%)", "Stop", "Target"],
                       tablefmt="rounded_grid" if HAS_TABULATE else "simple"))
    else:
        print(f"\n  {c('No open positions', Fore.YELLOW)}")

    # P&L stats
    pnl = status.get("pnl_stats", {})
    if pnl and pnl.get("total_trades"):
        print_section("CLOSED TRADE STATS")
        total  = pnl.get("total_trades", 0)
        wins   = pnl.get("winners", 0)
        losses = pnl.get("losers", 0)
        wr     = (wins / total * 100) if total > 0 else 0
        print(f"  Total trades:  {total}")
        print(f"  Win rate:      {c(f'{wr:.1f}%', Fore.GREEN if wr >= 50 else Fore.RED)}")
        print(f"  Total P&L:     {fmt_pnl(pnl.get('total_pnl'))}")
        print(f"  Avg trade:     {fmt_pnl(pnl.get('avg_pnl'))}")
        print(f"  Best trade:    {fmt_pnl(pnl.get('best_trade'))}")
        print(f"  Worst trade:   {fmt_pnl(pnl.get('worst_trade'))}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-symbol analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_single(engine, symbol: str):
    print_section(f"ANALYSING: {symbol.upper()}")
    print(f"  Running all 10 techniques…\n")

    from signal_engine import _fetch_data_bundle
    from techniques import ALL_TECHNIQUES

    data = _fetch_data_bundle(symbol)
    results = []
    rows = []
    for TechClass in ALL_TECHNIQUES:
        tech   = TechClass()
        result = tech.analyse(symbol, data)
        results.append(result)
        sig_color = (Fore.GREEN if result.signal == "BUY"
                     else Fore.RED if result.signal == "SELL"
                     else Fore.YELLOW)
        verdict = result.reasoning.get("verdict", "")[:55]
        rows.append([
            result.name,
            c(result.signal, sig_color),
            f"{result.score:.0f}",
            f"{result.confidence:.0%}",
            "✓" if result.applicable else "–",
            verdict,
        ])

    print(tabulate(rows,
                   headers=["Technique", "Signal", "Score", "Conf", "App.", "Verdict"],
                   tablefmt="rounded_grid" if HAS_TABULATE else "simple"))

    applicable = [r for r in results if r.applicable]
    buys  = sum(1 for r in applicable if r.signal == "BUY")
    sells = sum(1 for r in applicable if r.signal == "SELL")
    print(f"\n  Votes: {c(f'{buys} BUY', Fore.GREEN)} / "
          f"{c(f'{sells} SELL', Fore.RED)} / "
          f"{len(applicable)-buys-sells} NEUTRAL "
          f"(threshold: {config.MIN_SIGNALS_TO_TRADE})")


# ─────────────────────────────────────────────────────────────────────────────
# Continuous loop
# ─────────────────────────────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n  {c('Shutdown signal received — finishing current cycle…', Fore.YELLOW)}")


def run_bot(engine, interval_minutes: int = None):
    global _shutdown_requested
    _shutdown_requested = False

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    interval = (interval_minutes or config.SCAN_INTERVAL) * 60

    print(f"\n{c('Bot started', Fore.GREEN)} | mode={engine.mode.upper()} | "
          f"scan every {interval//60} min\n"
          f"Press Ctrl+C to stop gracefully.\n")

    cycle = 0
    while not _shutdown_requested:
        cycle += 1
        print(f"\n{c(f'─── SCAN #{cycle} ───', Fore.CYAN)} "
              f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

        try:
            summary = engine.run_cycle()
            print(f"  Signals found:  {summary['signals_found']}")
            print(f"  Trades executed: {c(str(summary['trades_executed']), Fore.GREEN)}")
            print(f"  Trades skipped:  {summary['trades_skipped']}")
            print(f"  Positions closed: {summary['positions_closed']}")

            if summary.get("errors"):
                print(f"  {c('Errors:', Fore.RED)} {len(summary['errors'])}")

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)

        if _shutdown_requested:
            break

        print_status(engine)
        print(f"\n  Next scan in {interval//60} minutes… "
              f"({datetime.now().strftime('%H:%M:%S')})")

        elapsed = 0
        while elapsed < interval and not _shutdown_requested:
            time.sleep(min(1, interval - elapsed))
            elapsed += 1

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    print(f"\n{c('Bot stopped gracefully.', Fore.GREEN)}")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive menu
# ─────────────────────────────────────────────────────────────────────────────

def close_trade_menu(engine):
    """Interactive menu to manually close one or more open positions."""
    import database as db
    positions = db.get_positions(engine.mode)
    if not positions:
        print(f"\n  {c('No open positions to close.', Fore.YELLOW)}")
        return

    print_section("CLOSE POSITIONS MANUALLY")
    rows = []
    for i, pos in enumerate(positions, 1):
        ep  = pos["entry_price"]
        cp  = pos["current_price"]
        qty = pos["quantity"]
        pnl = (cp - ep) * qty
        rows.append([
            i,
            pos["symbol"],
            f"{qty:.6g}",
            f"${ep:.4f}",
            f"${cp:.4f}",
            fmt_pnl(pnl),
        ])
    print(tabulate(rows,
                   headers=["#", "Symbol", "Qty", "Entry", "Current", "Unrealised P&L"],
                   tablefmt="rounded_grid" if HAS_TABULATE else "simple"))
    print(f"\n  Enter position number(s) to close, 'all' to close all, or 0 to cancel.")

    try:
        raw = input(f"  {c('Choice:', Fore.CYAN)} ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return

    if raw == "0" or raw == "":
        return

    if raw == "all":
        targets = positions
    else:
        indices = []
        for part in raw.replace(",", " ").split():
            try:
                idx = int(part)
                if 1 <= idx <= len(positions):
                    indices.append(idx - 1)
                else:
                    print(f"  {c(f'Invalid number: {part}', Fore.RED)}")
            except ValueError:
                print(f"  {c(f'Not a number: {part}', Fore.RED)}")
        targets = [positions[i] for i in indices]

    if not targets:
        return

    symbols = [p["symbol"] for p in targets]
    confirm = input(f"  Close {c(', '.join(symbols), Fore.YELLOW)}? (yes/no): ").strip().lower()
    if confirm not in ("yes", "y"):
        print(f"  {c('Cancelled.', Fore.YELLOW)}")
        return

    for pos in targets:
        symbol = pos["symbol"]
        if engine.mode == "paper" and engine.paper_trader:
            pnl = engine.paper_trader.execute_sell(symbol, reason="manual")
            if pnl is not None:
                print(f"  {c('Closed', Fore.GREEN)} {symbol} | P&L: {fmt_pnl(pnl)}")
            else:
                print(f"  {c(f'Failed to close {symbol}', Fore.RED)}")
        else:
            from data.market_data import get_current_price
            price = get_current_price(symbol)
            if price:
                engine._close_live_position(pos, price, reason="manual")
                print(f"  {c('Closed', Fore.GREEN)} {symbol} @ ${price:.4f}")
            else:
                print(f"  {c(f'Could not get price for {symbol}', Fore.RED)}")


def interactive_menu(engine, interval_minutes: int = None):
    while True:
        print_section("MAIN MENU")
        print("  [1] Start auto-trading loop")
        print("  [2] Run one scan now")
        print("  [3] View current status & positions")
        print("  [4] Analyse a single symbol")
        print("  [5] View recent trades")
        print("  [6] Close positions manually")
        print("  [7] Switch mode (Paper ↔ Live)")
        print("  [0] Exit")

        try:
            choice = input(f"\n  {c('Select option:', Fore.CYAN)} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting…")
            break

        if choice == "1":
            try:
                run_bot(engine, interval_minutes=interval_minutes)
            except KeyboardInterrupt:
                print(f"\n{c('Loop stopped.', Fore.YELLOW)}")

        elif choice == "2":
            print(f"\n{c('Running scan…', Fore.CYAN)}")
            summary = engine.run_cycle()
            print(f"  Done: {summary['trades_executed']} trades executed")

        elif choice == "3":
            print_status(engine)

        elif choice == "4":
            symbol = input("  Enter symbol (e.g. AAPL): ").strip().upper()
            if symbol:
                analyse_single(engine, symbol)

        elif choice == "5":
            import database as db
            trades = db.get_trade_history(engine.mode, limit=20)
            if trades:
                print_section("RECENT TRADES")
                rows = [[
                    t["timestamp"][:16],
                    t["symbol"],
                    t["side"],
                    f"{t['quantity']:.6g}",
                    f"${t['entry_price']:.4f}",
                    f"${t['exit_price']:.4f}" if t.get("exit_price") else "open",
                    fmt_pnl(t.get("pnl")),
                    t["status"],
                ] for t in trades]
                print(tabulate(rows,
                               headers=["Time", "Symbol", "Side", "Qty",
                                        "Entry", "Exit", "P&L", "Status"],
                               tablefmt="rounded_grid" if HAS_TABULATE else "simple"))
            else:
                print(f"  {c('No trades yet.', Fore.YELLOW)}")

        elif choice == "6":
            close_trade_menu(engine)

        elif choice == "7":
            current = engine.mode
            new_mode = "live" if current == "paper" else "paper"
            confirm = input(f"  Switch from {current.upper()} to "
                            f"{new_mode.upper()}? (yes/no): ").strip().lower()
            if confirm == "yes":
                os.environ["TRADING_MODE"] = new_mode
                import importlib
                importlib.reload(config)
                from trading_engine import TradingEngine
                engine = TradingEngine(mode=new_mode)
                print(f"  {c(f'Switched to {new_mode.upper()} mode', Fore.GREEN)}")

        elif choice == "0":
            print(f"\n{c('Goodbye!', Fore.CYAN)}")
            break
        else:
            print(f"  {c('Invalid option.', Fore.RED)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def startup_wizard() -> tuple[bool, float, int]:
    """
    Interactive startup prompts shown when the bot is launched with no CLI flags.
    Returns (reset: bool, capital: float, interval_minutes: int).
    """
    print_section("STARTUP SETUP")

    # ── 1. Reset portfolio? ───────────────────────────────────────────────────
    while True:
        ans = input(f"  {c('Reset portfolio?', Fore.CYAN)} "
                    f"[y/N]: ").strip().lower()
        if ans in ("", "n", "no"):
            reset = False
            break
        if ans in ("y", "yes"):
            reset = True
            break
        print(f"  {c('Please enter y or n.', Fore.RED)}")

    # ── 2. Starting capital ───────────────────────────────────────────────────
    default_capital = config.PAPER_CAPITAL
    while True:
        raw = input(f"  {c('Starting capital', Fore.CYAN)} "
                    f"[${default_capital:,.0f}]: ").strip()
        if raw == "":
            capital = default_capital
            break
        try:
            capital = float(raw.replace(",", "").replace("$", ""))
            if capital <= 0:
                raise ValueError
            break
        except ValueError:
            print(f"  {c('Enter a positive number, e.g. 10000', Fore.RED)}")

    # ── 3. Scan interval ──────────────────────────────────────────────────────
    default_interval = config.SCAN_INTERVAL
    while True:
        raw = input(f"  {c('Scan interval (minutes)', Fore.CYAN)} "
                    f"[{default_interval}]: ").strip()
        if raw == "":
            interval = default_interval
            break
        try:
            interval = int(raw)
            if interval <= 0:
                raise ValueError
            break
        except ValueError:
            print(f"  {c('Enter a positive integer, e.g. 30', Fore.RED)}")

    return reset, capital, interval


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Trading Bot — 10-technique market analyser"
    )
    parser.add_argument("--mode",     choices=["paper", "live"],
                        help="Override trading mode")
    parser.add_argument("--capital",  type=float,
                        help="Starting capital for paper trading, e.g. --capital 25000")
    parser.add_argument("--reset",    action="store_true",
                        help="Reset portfolio (clears positions, trades, history) before starting")
    parser.add_argument("--scan",     action="store_true",
                        help="Run one scan cycle and exit")
    parser.add_argument("--status",   action="store_true",
                        help="Print current status and exit")
    parser.add_argument("--symbol",   type=str,
                        help="Analyse a single symbol and exit")
    parser.add_argument("--interval", type=int,
                        help="Scan interval in minutes (overrides config)")
    parser.add_argument("--run",       action="store_true",
                        help="Skip menu and start trading loop immediately (for server/service use)")
    parser.add_argument("--portfolio", type=str, default=None,
                        help="Portfolio name — each name gets its own isolated database "
                             "(e.g. --portfolio aggressive). Defaults to 'default'.")
    args = parser.parse_args()

    import importlib
    if args.mode:
        os.environ["TRADING_MODE"] = args.mode
    if args.capital:
        os.environ["PAPER_CAPITAL"] = str(args.capital)
    if args.portfolio:
        # Each portfolio name gets its own SQLite file
        safe_name = args.portfolio.replace(" ", "_").lower()
        os.environ["DATABASE_PATH"] = f"trading_bot_{safe_name}.db"
    importlib.reload(config)

    print_banner()

    # ── Interactive wizard — only when launched bare (no action flags) ────────
    non_interactive = args.scan or args.status or args.symbol or args.run
    interval_minutes = args.interval

    if not non_interactive and not args.capital and not args.reset:
        try:
            do_reset, capital, interval_minutes = startup_wizard()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{c('Cancelled.', Fore.YELLOW)}")
            return
        os.environ["PAPER_CAPITAL"] = str(capital)
        importlib.reload(config)
        args.reset = do_reset

    portfolio_name = args.portfolio or "default"
    print(f"\n  Portfolio: {c(portfolio_name, Fore.MAGENTA)}")
    print(f"  Database:  {config.DATABASE_PATH}")
    print(f"  Mode: {c(config.TRADING_MODE.upper(), Fore.CYAN)}")
    print(f"  Capital: ${config.PAPER_CAPITAL:,.0f}")
    print(f"  Risk/trade: {config.RISK_PER_TRADE:.0%}")
    print(f"  Min signals: {config.MIN_SIGNALS_TO_TRADE}/10")
    if interval_minutes:
        print(f"  Scan interval: {interval_minutes} min")
    print()

    # Initialise database
    import database as db
    db.init_db()

    # Reset if requested
    if args.reset:
        db.reset_portfolio(config.PAPER_CAPITAL)
        print(f"  {c(f'Portfolio reset → ${config.PAPER_CAPITAL:,.0f}', Fore.YELLOW)}\n")

    # Create engine
    from trading_engine import TradingEngine
    engine = TradingEngine()

    # Route based on args
    if args.status:
        print_status(engine)
    elif args.symbol:
        analyse_single(engine, args.symbol.upper())
    elif args.scan:
        print(f"{c('Running single scan…', Fore.CYAN)}")
        summary = engine.run_cycle()
        print_status(engine)
    elif args.run:
        run_bot(engine, interval_minutes=interval_minutes)
    else:
        interactive_menu(engine, interval_minutes=interval_minutes)


if __name__ == "__main__":
    main()

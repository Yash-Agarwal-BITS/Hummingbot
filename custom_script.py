import logging
from decimal import Decimal
from typing import Dict, List, Optional
import pandas as pd  # Import pandas for status formatting

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

class ImprovedCEXMM(ScriptStrategyBase):
    """
    Improved Custom Market Making Strategy for a CEX

    Combines features from previous versions and the user-provided EnhancedPMM:
      - Volatility-based spread adjustment (NATR + base spread)
      - Trend analysis via RSI (shifting reference price, with threshold halts)
      - Inventory risk management (target ratio, min/max limits, emergency spread widening)
    """

    # --- Core Strategy Parameters ---
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"  # Use your actual exchange/paper trade connector
    order_refresh_time = 15  # In seconds
    order_amount = Decimal("0.01")  # Base asset amount per order
    price_source = PriceType.MidPrice  # Or PriceType.LastTrade, etc.
    create_timestamp = 0  # Internal timer variable

    # Split trading pair into base and quote
    base, quote = trading_pair.split('-')

    # --- Candles Configuration ---
    candle_exchange = "binance"  # Source exchange for candle data (e.g., "binance", "kucoin")
    candles_interval = "1m"  # Candle interval (e.g., "1m", "5m", "1h")
    candles_length = 30  # Number of periods for TA indicators (NATR, RSI)
    max_records = 1000  # Max candle records to store (adjust based on memory)

    # --- Volatility & Spread Parameters ---
    base_bid_spread = Decimal("0.0001")  # Minimum bid spread (adds to NATR adjustment), e.g., 0.01%
    base_ask_spread = Decimal("0.0001")  # Minimum ask spread (adds to NATR adjustment), e.g., 0.01%
    bid_spread_scalar = Decimal("120")  # Multiplier for NATR -> bid spread adjustment
    ask_spread_scalar = Decimal("80")   # Multiplier for NATR -> ask spread adjustment

    # --- Trend Adjustment Parameters ---
    rsi_oversold = 30  # RSI level below which trading might halt (risk control)
    rsi_overbought = 70  # RSI level above which trading might halt (risk control)
    trend_scalar = Decimal("-0.00005")  # Multiplier for RSI deviation -> price shift percentage (TUNE THIS)
                                      # Negative value means high RSI (overbought) pushes price down
    max_trend_shift_pct = Decimal("0.001")  # Max allowed price shift due to trend (e.g., 0.1%) (TUNE THIS)

    # --- Inventory Management Parameters ---
    target_inventory_ratio = Decimal("0.5")   # Target base asset ratio (e.g., 0.5 = 50% of total value)
    min_inventory_ratio = Decimal("0.35")  # Lower bound for base asset ratio before emergency measures (TUNE THIS)
    max_inventory_ratio = Decimal("0.65")  # Upper bound for base asset ratio before emergency measures (TUNE THIS)
    inventory_scalar = Decimal("0.1")  # Multiplier for inventory deviation -> price shift percentage (TUNE THIS CAREFULLY!)
    max_inventory_shift_pct = Decimal("0.005")  # Max allowed price shift due to inventory (e.g., 0.5%) (TUNE THIS)
    emergency_spread_widen_factor = Decimal("2.0")  # Factor to widen spreads when inventory limits breached (TUNE THIS)
    # NEW: Maximum allowed spreads in emergency mode to prevent orders from being too far apart
    emergency_max_bid_spread = Decimal("0.005")  # Maximum bid spread (e.g., 0.5%)
    emergency_max_ask_spread = Decimal("0.005")  # Maximum ask spread (e.g., 0.5%)

    # --- State Variables (Internal) ---
    reference_price = Decimal("1")  # Price after trend and inventory adjustments
    bid_spread = Decimal("0")  # Current dynamic bid spread
    ask_spread = Decimal("0")  # Current dynamic ask spread
    trend_multiplier_pct = Decimal("0")  # Current price adjustment factor from trend
    inventory_multiplier_pct = Decimal("0")  # Current price adjustment factor from inventory
    last_rsi = Decimal("50")  # Store last RSI for risk checks
    inventory_emergency_mode = False  # Flag if inventory limits are breached

    # Initialize candles feed
    candles = CandlesFactory.get_candle(
        CandlesConfig(
            connector=candle_exchange,
            trading_pair=trading_pair,
            interval=candles_interval,
            max_records=max_records
        )
    )

    # Define markets
    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()
        self.logger().info("ImprovedCEXMM strategy started with parameters:")
        self.logger().info(f"  Inventory Target: {self.target_inventory_ratio:.1%}, Limits: [{self.min_inventory_ratio:.1%}-{self.max_inventory_ratio:.1%}]")
        self.logger().info(f"  Inventory Scalar: {self.inventory_scalar}, Max Shift: {self.max_inventory_shift_pct:.4%}")
        self.logger().info(f"  Trend Scalar: {self.trend_scalar}, Max Shift: {self.max_trend_shift_pct:.4%}, RSI Limits: [{self.rsi_oversold}-{self.rsi_overbought}]")
        self.logger().info(f"  Emergency Spread Factor: {self.emergency_spread_widen_factor}")

    def on_stop(self):
        self.candles.stop()
        self.logger().info("ImprovedCEXMM strategy stopped.")

    def on_tick(self):
        """Main strategy loop executed periodically."""
        if self.create_timestamp <= self.current_timestamp:
            # 1. Cancel previous orders
            self.cancel_all_orders()

            # 2. Update dynamic strategy parameters
            params_updated = self.update_strategy_parameters()
            if not params_updated:  # Skip cycle if candles/indicators aren't ready or price is invalid
                self.logger().warning("Strategy parameters not updated. Skipping cycle.")
                self.create_timestamp = self.order_refresh_time + self.current_timestamp
                return

            # 3. Create new order proposals
            proposal: List[OrderCandidate] = self.create_proposal()

            # 4. Apply risk management rules
            proposal_risk_adjusted: List[OrderCandidate] = self.apply_risk_management(proposal)

            # 5. Adjust proposal to available budget
            proposal_budget_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal_risk_adjusted)

            # 6. Place the final orders
            self.place_orders(proposal_budget_adjusted)

            # 7. Set timer for next execution
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

    def get_candle_indicators(self):
        """Calculates NATR and RSI, appending them to the candle DataFrame."""
        candles_df = self.candles.candles_df
        if candles_df is None or candles_df.empty:
            self.logger().warning("Candle DataFrame is empty or None.")
            return None
        try:
            # Calculate NATR (Normalized Average True Range) for volatility
            candles_df.ta.natr(length=self.candles_length, scalar=1, append=True)
            # Calculate RSI (Relative Strength Index) for trend
            candles_df.ta.rsi(length=self.candles_length, append=True)
            return candles_df
        except Exception as e:
            self.logger().error(f"Error calculating TA indicators: {e}", exc_info=True)
            return None

    def update_strategy_parameters(self) -> bool:
        """
        Updates dynamic parameters based on market conditions (volatility, trend, inventory).
        Returns True if successful, False otherwise (e.g., candles not ready).
        """
        if not self.candles.ready:
            self.logger().warning("Candles not ready yet. Skipping parameter update.")
            return False

        candles_df = self.get_candle_indicators()
        if candles_df is None or candles_df.empty or f"NATR_{self.candles_length}" not in candles_df.columns or f"RSI_{self.candles_length}" not in candles_df.columns:
            self.logger().warning("Required indicators not available in candles data. Skipping parameter update.")
            return False

        # Check for NaN values in the last row for required indicators
        last_row = candles_df.iloc[-1]
        if last_row[[f"NATR_{self.candles_length}", f"RSI_{self.candles_length}"]].isnull().any():
            self.logger().warning("NaN values detected in latest indicators (NATR/RSI). Skipping parameter update.")
            return False

        mid_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        if mid_price <= 0 or not mid_price.is_finite():
            self.logger().warning(f"Invalid mid price ({mid_price}). Skipping parameter update.")
            return False

        # --- 1. Volatility Adjustment (Spreads) ---
        latest_natr = Decimal(str(last_row[f"NATR_{self.candles_length}"]))
        # Calculate dynamic spreads based on NATR and base spreads
        self.bid_spread = self.base_bid_spread + (latest_natr * self.bid_spread_scalar)
        self.ask_spread = self.base_ask_spread + (latest_natr * self.ask_spread_scalar)

        # --- 2. Trend Adjustment (RSI Price Shift) ---
        self.last_rsi = Decimal(str(last_row[f"RSI_{self.candles_length}"]))
        # Calculate trend shift percentage based on RSI deviation from 50
        # trend_scalar controls magnitude and direction (-ve means high RSI pushes price down)
        trend_shift_pct_raw = (self.last_rsi - Decimal("50")) / Decimal("50") * self.trend_scalar
        # Clamp the trend shift to the maximum allowed percentage
        self.trend_multiplier_pct = max(min(trend_shift_pct_raw, self.max_trend_shift_pct), -self.max_trend_shift_pct)

        # --- 3. Inventory Adjustment (Price Shift & Emergency Spreads) ---
        base_balance = self.connectors[self.exchange].get_balance(self.base)
        quote_balance = self.connectors[self.exchange].get_balance(self.quote)
        if base_balance is None or quote_balance is None:
            self.logger().warning("Could not fetch balances. Skipping inventory adjustment.")
            return False  # Cannot proceed without balances

        base_value_in_quote = base_balance * mid_price
        total_value = base_value_in_quote + quote_balance

        current_ratio = (base_value_in_quote / total_value) if total_value > 0 else self.target_inventory_ratio

        # Calculate inventory shift percentage based on deviation from target
        # Positive delta = overweight base -> shift price down (negative multiplier)
        delta_ratio = current_ratio - self.target_inventory_ratio
        # Calculate raw shift based on deviation and scalar
        inventory_shift_pct_raw = -delta_ratio * self.inventory_scalar  # Negative sign adjusts price correctly
        # Clamp the final shift percentage
        self.inventory_multiplier_pct = max(min(inventory_shift_pct_raw, self.max_inventory_shift_pct), -self.max_inventory_shift_pct)

        # Check inventory limits for emergency measures
        self.inventory_emergency_mode = False
        if current_ratio > self.max_inventory_ratio or current_ratio < self.min_inventory_ratio:
            self.inventory_emergency_mode = True
            self.logger().warning(
                f"Inventory ratio {current_ratio:.2%} outside limits [{self.min_inventory_ratio:.0%}-{self.max_inventory_ratio:.0%}]. "
                "Engaging emergency spread widening."
            )
            # Widen spreads by the factor
            self.bid_spread *= self.emergency_spread_widen_factor
            self.ask_spread *= self.emergency_spread_widen_factor

            # Limit the spread widening to the maximum allowed values
            if self.bid_spread > self.emergency_max_bid_spread:
                self.bid_spread = self.emergency_max_bid_spread
            if self.ask_spread > self.emergency_max_ask_spread:
                self.ask_spread = self.emergency_max_ask_spread

        # --- 4. Final Reference Price Calculation ---
        # Apply trend and inventory shifts multiplicatively to the original mid price
        self.reference_price = mid_price * (Decimal("1") + self.trend_multiplier_pct) * (Decimal("1") + self.inventory_multiplier_pct)

        return True  # Parameters updated successfully

    def create_proposal(self) -> List[OrderCandidate]:
        """Creates buy and sell order proposals around the adjusted reference price."""
        ref_price = self.reference_price
        buy_price_candidate = ref_price * (Decimal("1") - self.bid_spread)
        sell_price_candidate = ref_price * (Decimal("1") + self.ask_spread)

        # Get best prices from order book to avoid crossing the spread
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)

        # Ensure we have valid best bid/ask prices
        if best_bid <= 0 or not best_bid.is_finite() or best_ask <= 0 or not best_ask.is_finite():
            self.logger().warning(f"Invalid best bid/ask prices ({best_bid}/{best_ask}). Cannot create proposal.")
            return []

        # Ensure orders don't cross the best market prices
        # Buy order price should be less than or equal to best bid
        # Sell order price should be greater than or equal to best ask
        final_buy_price = min(buy_price_candidate, best_bid)
        final_sell_price = max(sell_price_candidate, best_ask)

        # Ensure buy price < sell price after adjustments
        if final_buy_price >= final_sell_price:
            self.logger().warning(f"Calculated buy price ({final_buy_price}) >= sell price ({final_sell_price}). Skipping proposal.")
            # Attempt to slightly adjust to prevent overlap - might need more sophisticated logic
            final_buy_price = ref_price * (Decimal("1") - self.ask_spread)  # Use ask spread for buy if overlapping
            final_sell_price = ref_price * (Decimal("1") + self.bid_spread)  # Use bid spread for sell if overlapping
            if final_buy_price >= final_sell_price:  # Still overlapping, skip
                self.logger().error("Buy price still >= sell price after attempted adjustment. Cannot create proposal.")
                return []

        buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                   order_side=TradeType.BUY, amount=self.order_amount, price=final_buy_price)
        sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=self.order_amount, price=final_sell_price)
        return [buy_order, sell_order]

    def apply_risk_management(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """
        Applies risk checks:
        - Halts trading if RSI is in extreme zones (oversold/overbought).
        Returns the original proposal or an empty list if risks are triggered.
        """
        # Halt trading if RSI is extreme (from EnhancedPMM)
        if self.last_rsi < self.rsi_oversold or self.last_rsi > self.rsi_overbought:
            self.logger().warning(f"RSI ({self.last_rsi:.2f}) outside safe range ({self.rsi_oversold}-{self.rsi_overbought}). Halting order placement.")
            return []  # Return empty list to prevent orders


        return proposal  # Pass through if no risks triggered

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjusts order candidates to fit available budget."""
        if not proposal:  # If risk management returned empty list
            return []
        try:
            adjusted_proposal = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
            return adjusted_proposal
        except Exception as e:
            self.logger().error(f"Error adjusting proposal to budget: {e}", exc_info=True)
            return []  # Return empty list on error

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Places the final adjusted order proposals."""
        if not proposal:
            self.logger().info("No valid order proposals to place.")
            return

        for order in proposal:
            # Log placement attempt
            self.logger().info(f"Placing {order.order_side.name} order: {order.amount} {order.trading_pair} @ {order.price:.6f}")
            self.place_order(connector_name=self.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        """Helper function to place buy or sell orders with error handling."""
        try:
            if order.order_side == TradeType.SELL:
                self.sell(connector_name=connector_name, trading_pair=order.trading_pair,
                          amount=order.amount, order_type=order.order_type, price=order.price)
            elif order.order_side == TradeType.BUY:
                self.buy(connector_name=connector_name, trading_pair=order.trading_pair,
                         amount=order.amount, order_type=order.order_type, price=order.price)
        except Exception as e:
            # Log the error but allow the strategy to continue
            self.logger().error(f"Failed to place {order.order_side.name} order for {order.trading_pair} at {order.price}: {e}", exc_info=True)

    def cancel_all_orders(self):
        """Cancels all active orders on the specified exchange."""
        try:
            active_orders = self.get_active_orders(connector_name=self.exchange)
            if active_orders:
                self.logger().info(f"Cancelling {len(active_orders)} active orders...")
                for order in active_orders:
                    self.cancel(self.exchange, order.trading_pair, order.client_order_id)
        except Exception as e:
            self.logger().error(f"Error cancelling orders: {e}", exc_info=True)

    def did_fill_order(self, event: OrderFilledEvent):
        """Logs filled order events."""
        msg = (f"FILLED: {event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} @ {round(event.price, 4)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)  # Sends notification to Hummingbot client

    def format_status(self) -> str:
        """Returns a detailed status string for the bot UI."""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        debug_lines = []  # For less critical info

        # --- General Status ---
        lines.append(f"Strategy: {type(self).__name__}")
        if self.inventory_emergency_mode:
            warning_lines.append("!!! INVENTORY EMERGENCY MODE ACTIVE - SPREADS WIDENED !!!")
        if not self.candles.ready:
            warning_lines.append("!!! CANDLES NOT READY - PARAMETERS MAY BE STALE !!!")

        # Safely get market prices
        mid_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)

        # --- Market Info ---
        lines.append("")
        lines.append(f"Market: {self.exchange} | {self.trading_pair}")
        if mid_price > 0 and best_bid > 0 and best_ask > 0:
            lines.append(f"  Mid Price: {mid_price:.6f} | Best Bid: {best_bid:.6f} | Best Ask: {best_ask:.6f}")
        else:
            warning_lines.append("!!! Market prices seem invalid !!!")
            lines.append(f"  Mid Price: {mid_price} | Best Bid: {best_bid} | Best Ask: {best_ask}")

        # --- Pricing Logic ---
        lines.append("")
        lines.append("Pricing Logic:")
        lines.append(f"  Base Price ({self.price_source.name}): {mid_price:.6f}")
        lines.append(f"  Trend Shift (RSI: {self.last_rsi:.2f}): {self.trend_multiplier_pct:+.4%} (Max: +/-{self.max_trend_shift_pct:.4%})")
        lines.append(f"  Inventory Shift: {self.inventory_multiplier_pct:+.4%} (Max: +/-{self.max_inventory_shift_pct:.4%})")
        lines.append(f"  ==> Adjusted Reference Price: {self.reference_price:.6f}")

        # --- Spreads & Orders ---
        lines.append("")
        lines.append("Spreads & Orders:")
        lines.append(f"  Base Spreads (Bid/Ask): {self.base_bid_spread:.4%} / {self.base_ask_spread:.4%}")
        lines.append(f"  Dynamic Spreads (Bid/Ask): {self.bid_spread:.4%} / {self.ask_spread:.4%}" + (" (EMERGENCY WIDENED)" if self.inventory_emergency_mode else ""))
        # Recalculate potential final prices for display consistency
        buy_price_calc = self.reference_price * (Decimal("1") - self.bid_spread)
        sell_price_calc = self.reference_price * (Decimal("1") + self.ask_spread)
        if best_bid > 0 and best_ask > 0:  # Only show final calc if market prices are valid
            final_buy_disp = min(buy_price_calc, best_bid)
            final_sell_disp = max(sell_price_calc, best_ask)
            lines.append(f"  ==> Calculated Buy: {final_buy_disp:.6f} | Calculated Sell: {final_sell_disp:.6f}")
        else:
            lines.append("  ==> Calculated Prices: (Market prices invalid)")

        lines.append(f"  Order Amount: {self.order_amount} {self.base}")

        # --- Inventory ---
        lines.append("")
        lines.append("Inventory:")
        base_balance = self.connectors[self.exchange].get_balance(self.base)
        quote_balance = self.connectors[self.exchange].get_balance(self.quote)
        if base_balance is not None and quote_balance is not None and mid_price > 0:
            base_value = base_balance * mid_price
            total_value = base_value + quote_balance
            current_ratio = (base_value / total_value) if total_value > 0 else Decimal("0")
            lines.append(f"  Target: {self.target_inventory_ratio:.1%} | Current: {current_ratio:.1%}")
            lines.append(f"  Limits: [{self.min_inventory_ratio:.1%}-{self.max_inventory_ratio:.1%}]")
            lines.append(f"  Balances: {base_balance:.4f} {self.base} | {quote_balance:.2f} {self.quote}")
        else:
            lines.append("  Balances: Error fetching or calculating inventory.")

        # --- Active Orders ---
        lines.append("")
        lines.append("Active Orders:")
        try:
            orders_df = self.active_orders_df()
            if not orders_df.empty:
                # Limit columns displayed just because
                orders_disp = orders_df[["Exchange", "Market", "Side", "Price", "Amount", "Age"]]
                lines.extend(["    " + line for line in orders_disp.to_string(index=False).split("\n")])
            else:
                lines.append("  None")
        except Exception as e:
            lines.append(f"  Error fetching active orders: {e}")

        # Combine warnings and main status lines
        return "\n".join(warning_lines + lines + debug_lines)

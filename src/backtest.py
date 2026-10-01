from dataclasses import dataclass

import pandas as pd

FEE = 0.00125  # comisión por operación (entrada y salida), fijada por el lab


@dataclass(frozen=True)
class Position:
    """Posición abierta. side: +1 largo, -1 corto. qty en unidades de BTC."""

    side: int
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp

    @classmethod
    def from_atr(cls, side, qty, entry_price, atr, m, r, entry_time):
        """SL = entrada -/+ m*ATR y TP = entrada +/- r*m*ATR (según el lado)."""
        stop_dist = m * atr
        return cls(
            side,
            qty,
            entry_price,
            entry_price - side * stop_dist,
            entry_price + side * r * stop_dist,
            entry_time,
        )


def size_position(equity, price, stop_distance, risk_frac, fee=FEE) -> float:
    """Unidades a operar: el riesgo en dólares (risk_frac * equity) dividido entre la
    distancia al stop, limitado para que el nocional (con comisión) no exceda el
    capital, porque el lab no permite apalancamiento."""
    qty_risk = risk_frac * equity / stop_distance
    qty_cap = equity / (price * (1 + fee))
    return min(qty_risk, qty_cap)


def check_exit(pos: Position, bar_open, bar_high, bar_low):
    """Devuelve (motivo, precio de salida) si la barra cierra la posición, o None.

    Primero se revisan los gaps (si la barra abre más allá de un nivel, se llena a la
    apertura). Si el stop y el target caen dentro de la misma barra, gana el stop
    (convención conservadora)."""
    if pos.side == 1:
        if bar_open <= pos.stop_loss:
            return "stop_loss", bar_open
        if bar_open >= pos.take_profit:
            return "take_profit", bar_open
        if bar_low <= pos.stop_loss:
            return "stop_loss", pos.stop_loss
        if bar_high >= pos.take_profit:
            return "take_profit", pos.take_profit
    else:
        if bar_open >= pos.stop_loss:
            return "stop_loss", bar_open
        if bar_open <= pos.take_profit:
            return "take_profit", bar_open
        if bar_high >= pos.stop_loss:
            return "stop_loss", pos.stop_loss
        if bar_low <= pos.take_profit:
            return "take_profit", pos.take_profit
    return None


class Portfolio:
    """Estado explícito: efectivo, a lo más una posición abierta y bitácora de operaciones.

    Contabilidad simétrica: abrir un largo resta el nocional del efectivo y abrir un
    corto lo suma; la posición se valúa en side * qty * precio. La comisión se cobra
    en cada apertura y cada cierre."""

    def __init__(self, cash: float, fee: float = FEE):
        self.cash = cash
        self.fee = fee
        self.position: Position | None = None
        self.trades: list[dict] = []
        self.total_costs = 0.0
        self.traded_notional = 0.0
        self._entry_cost = 0.0

    def _trade_cash(self, side: int, qty: float, price: float, opening: bool) -> float:
        """Mueve el efectivo por una operación y devuelve la comisión pagada."""
        notional = qty * price
        cost = self.fee * notional
        direction = -side if opening else side
        self.cash += direction * notional - cost
        self.total_costs += cost
        self.traded_notional += notional
        return cost

    def open(self, pos: Position) -> None:
        if self.position is not None:
            raise RuntimeError("Ya hay una posición abierta")
        self._entry_cost = self._trade_cash(pos.side, pos.qty, pos.entry_price, True)
        self.position = pos

    def close(self, price: float, time: pd.Timestamp, reason: str) -> None:
        pos = self.position
        if pos is None:
            raise RuntimeError("No hay posición abierta")
        exit_cost = self._trade_cash(pos.side, pos.qty, price, False)
        self.trades.append(
            {
                "entry_time": pos.entry_time,
                "exit_time": time,
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "reason": reason,
                "pnl": pos.side * pos.qty * (price - pos.entry_price)
                - self._entry_cost
                - exit_cost,
            }
        )
        self.position = None

    def equity(self, mark_price: float) -> float:
        """Valor del portafolio: efectivo más el valor de la posición abierta."""
        pos_value = self.position.side * self.position.qty * mark_price if self.position else 0.0
        return self.cash + pos_value
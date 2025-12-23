"""Demonstrate custom logging during a fire simulation.

These log messages appear in the ``messages`` section of ``status_log.json`` and can
be paired with other log artifacts (cell/agent/action/prediction parquet files).
To try it, start a fire sim and select this file as the User Module.
"""

from embrs.base_classes.control_base import ControlClass
from embrs.fire_simulator.fire import FireSim


class CustomLogging(ControlClass):
    SUMMARY_INTERVAL_S = 15 * 60
    AREA_THRESHOLD_M2 = 5_000

    def __init__(self, fire: FireSim):
        fire.logger.log_message(
            "Custom Logging example running: status messages and sample prediction logging enabled."
        )

        self.operation_performed = False
        self.area_message_logged = False
        self.prediction_logged = False
        self._next_summary_time_s = fire.curr_time_s + self.SUMMARY_INTERVAL_S

    def process_state(self, fire: FireSim) -> None:
        if not self.operation_performed and fire.curr_time_h > 1:
            self.perform_some_operation(fire)

        self._log_area_threshold(fire)
        self._log_periodic_summary(fire)

        if fire.burning_cells and not self.prediction_logged:
            self._log_prediction_example(fire)
            self.prediction_logged = True

    def perform_some_operation(self, fire: FireSim):
        fire.logger.log_message(f"Some operation performed at {fire.curr_time_h:.2f} hours.")
        self.operation_performed = True

    def _log_area_threshold(self, fire: FireSim):
        burning_area = self._calc_burning_area_m2(fire)
        if burning_area > self.AREA_THRESHOLD_M2 and not self.area_message_logged:
            fire.logger.log_message(
                f"Burning area exceeded {self.AREA_THRESHOLD_M2:,.0f} m^2 "
                f"(current: {burning_area:,.0f} m^2)."
            )
            self.area_message_logged = True

    def _log_periodic_summary(self, fire: FireSim):
        if fire.curr_time_s < self._next_summary_time_s:
            return

        burning_cells = len(fire.burning_cells)
        frontier_cells = len(fire.frontier)
        avg_coord = self._format_avg_coord(fire)
        fire.logger.log_message(
            f"[t={fire.curr_time_m:.1f} min] burning={burning_cells} | frontier={frontier_cells} | "
            f"avg fire coord={avg_coord}"
        )
        fire.logger.flush()  # ensure the summary lands promptly in the log files
        self._next_summary_time_s += self.SUMMARY_INTERVAL_S

    def _log_prediction_example(self, fire: FireSim):
        anchor = fire.burning_cells[0]
        fire.curr_prediction = {anchor.id: (anchor.row, anchor.col)}
        fire.logger.log_message(
            "Queued a sample prediction entry (see prediction_logs.parquet for results)."
        )

    @staticmethod
    def _calc_burning_area_m2(fire: FireSim) -> float:
        if not fire.burning_cells:
            return 0.0
        cell_area = fire.burning_cells[0].cell_area
        return len(fire.burning_cells) * cell_area

    @staticmethod
    def _format_avg_coord(fire: FireSim) -> str:
        if not fire.burning_cells:
            return "n/a"
        x_avg, y_avg = fire.get_avg_fire_coord()
        return f"({x_avg:.1f} m, {y_avg:.1f} m)"

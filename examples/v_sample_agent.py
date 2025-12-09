"""Sample class showing how a custom agent class can be created using the AgentBase class.

There is nothing to run in this example, to see how it is used see the next two examples 'vi_controlled_burning.py
and 'vii_suppression.py'
"""

import numpy as np

from embrs.base_classes.agent_base import AgentBase
from embrs.models.fuel_models import FuelConstants as fc

class SampleAgent(AgentBase):
    def __init__(self, id: any, x: float, y: float, fire, label=None, marker='*', color='magenta'):
        super().__init__(id, x, y, label, marker=marker, color=color)

        # Define useful variables for agent class
        self.fire = fire
        self.task_list = []
        self.curr_task = None
        self.t_step = fire.time_step # seconds
        self.speed = 10 # m/s
        self.max_tasks = 0
        self.tasks_done = False

    def complete_tasks(self):
        if len(self.task_list) > 0:
            if self.curr_task is not None:

                # get current x and y
                x1 = self.x
                y1 = self.y

                # get x and y of next point
                x2 = self.curr_task[0][0]
                y2 = self.curr_task[0][1]

                # calculate total distance
                dx_total = x2 - x1
                dy_total = y2 - y1
                d_total = np.linalg.norm((dx_total, dy_total))

                # calculate incremental vector pointing at next point
                dir = np.arctan2(y2 - y1, x2 - x1)
                lin = self.t_step * self.speed

                # get dx and dy
                dx = lin * np.cos(dir)
                dy = lin * np.sin(dir)

                # check if dx, dy will overshoot the point
                if lin >= d_total:
                    # overshot so set x and y to goal position
                    self.x = x2
                    self.y = y2

                    if self.curr_task[1] == 'burn':
                        # Carry out burnout operation
                        self.fire.set_prescribed_fire_at_xy(self.x, self.y)
                        self.tasks_done = True

                    elif self.curr_task[1] == 'soak':
                        # Carry out suppression operation
                        cell = self.fire.get_cell_from_xy(self.x, self.y)
                        max_moisture = fc.dead_fuel_moisture_ext_table[cell.fuel_type.fuel_type]
                        self.fire.set_fuel_moisture_at_xy(self.x, self.y, max_moisture)
                        self.tasks_done = True

                    self.task_list.remove(self.curr_task)
                    self.curr_task = None

                else:
                    # Update agent position
                    self.x += dx
                    self.y += dy

            else:
                # Find next task based on which is closest
                min_dist = np.inf
                best_task = None
                for task in self.task_list:

                    dist = np.sqrt((self.x - task[0][0])**2 + (self.y - task[0][1])**2)

                    if dist < min_dist:
                        best_task = task

                self.curr_task = best_task
"""Ember lofting, transport, and spot fire ignition model.

Simulate the generation of firebrands (embers) from torching trees, their
ballistic transport through the atmosphere, and potential spot fire ignition
upon landing. Based on the Albini (1979) torching-tree firebrand model.

Classes:
    - Embers: Manage firebrand lofting, flight, and landing for a simulation.

References:
    Albini, F. A. (1979). Spot fire distance from burning trees — a
    predictive model. USDA Forest Service General Technical Report INT-56.
"""

from embrs.utilities.fire_util import CrownStatus, CanopySpecies, CellStates
from embrs.utilities.unit_conversions import *
from embrs.fire_simulator.cell import Cell
from typing import Callable, Tuple, Set

import numpy as np

class Embers:
    """Firebrand lofting, atmospheric transport, and spot fire ignition.

    Manage a collection of airborne embers, lofting them from torching
    cells, transporting them through the wind field, and igniting spot
    fires where they land on fuel cells.

    Attributes:
        ign_prob (float): Probability of spot fire ignition per firebrand
            (0 to 1).
        min_spot_dist_m (float): Minimum horizontal distance (meters) a
            firebrand must travel before it can ignite a spot fire.
        embers (list[dict]): Currently airborne firebrands, each a dict
            with keys 'x', 'y', 'diam', 'height', 'start_elev',
            'curr_time', 'elapsed'.
    """

    def __init__(self, ign_prob: float, species: int, dbh: float,
                 min_spot_dist: float, limits: Tuple[float, float],
                 get_cell_from_xy: Callable[[float, float], Cell]):
        """Initialize the ember model.

        Args:
            ign_prob (float): Spot fire ignition probability per firebrand
                (0 to 1).
            species (int): Canopy species ID (see ``CanopySpecies``).
            dbh (float): Diameter at breast height of representative trees
                (centimeters).
            min_spot_dist (float): Minimum spotting distance (meters).
            limits (Tuple[float, float]): ``(x_lim, y_lim)`` simulation
                domain bounds (meters).
            get_cell_from_xy (Callable): Callback to retrieve a Cell from
                spatial coordinates.

        Raises:
            ValueError: If ``species`` is not in ``CanopySpecies``.
        """
        # Call back injection
        self.get_cell_from_xy = get_cell_from_xy

        self.x_lim, self.y_lim = limits

        # Spotting Ignition Probability
        self.ign_prob = ign_prob

        # Minimum spotting distance (prevents excessive spotting that is going to burn imminently anyways)
        self.min_spot_dist_m = min_spot_dist

        # Load the canopy species properties
        self.A = CanopySpecies.properties

        if CanopySpecies.species_names.get(species) is None:
            raise ValueError(f"Species {species} not found in CanopySpecies.")

        self.treespecies = species
        self.dbh = dbh # cm

        self.embers = []

    def loft(self, cell: Cell, sim_time_m: float):
        """Loft firebrands from a torching cell.

        Compute plume characteristics, then attempt to loft up to 16
        embers of increasing diameter. Each passes a probabilistic check
        against ``ign_prob``. Successfully lofted embers are appended to
        ``self.embers``.

        Args:
            cell (Cell): The burning cell that is torching.
            sim_time_m (float): Current simulation time (minutes).

        Side Effects:
            Sets ``cell.lofted = True``. Appends ember dicts to
            ``self.embers``.
        """
        cell.lofted = True
        
        self.plume(cell)
        crown_height_ft = m_to_ft(cell.canopy_height)
        z_0 = crown_height_ft

        # Attempts to loft 16 embers of different types
        for count in range(1, 17):
            diameter = 0.005 * count

            # Only loft embers that pass probability check
            prob_spot = np.random.random() # [0, 1]
            if prob_spot > self.ign_prob:
                continue

            zf = self.torchheight(diameter, z_0)
            if zf > 0:
                # Appends to self.embers
                ember = {
                    'x': cell.x_pos,
                    'y': cell.y_pos,
                    'source_x': cell.x_pos,
                    'source_y': cell.y_pos,
                    'diam': diameter,
                    'height': ft_to_m(zf),
                    'start_elev': cell.elevation_m,
                    'curr_time': sim_time_m, # Times are in minutes
                    'elapsed': 0.0
                }
                self.embers.append(ember)

            else:
                break

    def partcalc(self, vowf: float, z: float) -> float:
        """Compute time residual for iterating maximum particle height.

        Used by ``torchheight`` to solve for the maximum lofting height of
        a firebrand as a function of particle diameter and flame height
        (Albini 1979).

        Args:
            vowf (float): Normalized terminal velocity ratio (dimensionless).
            z (float): Candidate particle height (feet).

        Returns:
            float: Time residual (dimensionless). Converges to zero at the
                correct maximum height.
        """
        # Calculates tT for iterating max particle height z as a function of particle diameter and flameheight (steady_height)
        # for torching trees only, function used by torcheight
        a = 5.963
        b = 4.563
        r2 = (b + z / self.steady_height) / a
        r = np.sqrt(r2)
        tT1 = a / 3.0 * (r * r2 - 1.0)
        pt8vowf = 0.8 * vowf
        Temp = abs((1.0 - pt8vowf) / (1.0 - pt8vowf * r))
        tp = a / (pt8vowf**3) * (np.log(Temp) - pt8vowf * (r - 1.0) - 0.5 * pt8vowf**2 * (r2 - 1.0))
        return tp - tT1

    def torchheight(self, diameter: float, z_0: float) -> float:
        """Compute maximum lofting height for a firebrand (Albini 1979).

        Iteratively solve for the height a firebrand of given diameter
        can reach from a torching tree, accounting for plume buoyancy and
        particle terminal velocity.

        Args:
            diameter (float): Firebrand diameter (feet).
            z_0 (float): Crown height / initial release height (feet).

        Returns:
            float: Maximum lofting height (feet), or -1.0 if the firebrand
                cannot be lofted.
        """
        # Albini 1979 torching trees model
        vowf = 40.0 * np.sqrt(diameter / self.steady_height)
        if vowf < 1.0:
            zo_ratio = np.sqrt(z_0 / self.steady_height)
            if zo_ratio > vowf:
                Temp = abs((1.0 - vowf) / (zo_ratio - vowf))
                tf = 1.0 - zo_ratio + vowf * np.log(Temp)
                tt = 0.2 * vowf * (1.0 + vowf * np.log(1.0 + 1.0 / (1.0 - vowf)))
                aT = self.duration + 1.2 - tf - tt
                if aT >= 0.0:
                    inc = 2.0
                    inc2 = 1.0
                    z = inc * self.steady_height
                    tT = self.partcalc(vowf, z)
                    while abs(tT - aT) > 0.01:
                        if tT == aT:
                            break
                        else:
                            if tT < aT:
                                inc += inc2
                            else:
                                inc -= inc2
                                inc2 /= 2.0
                                inc += inc2
                        z = inc * self.steady_height
                        tT = self.partcalc(vowf, z)
                    return z
        return -1.0

    def calc_front_dist(self, cell: Cell) -> float:
        """Estimate the fire front length at a cell.

        Approximate the frontal distance as the cell side length times the
        number of burnable neighbors.

        Args:
            cell (Cell): Cell to evaluate.

        Returns:
            float: Estimated fire front length (meters).
        """
        a = cell.cell_size
        num_neighbors = len(cell.burnable_neighbors)

        dist = a * num_neighbors
        return dist

    def plume(self, cell: Cell):
        """Compute plume steady-state height and flame duration for a cell.

        Determine the plume characteristics based on crown fire status,
        crown fraction burned, canopy cover, and species properties. The
        number of torching trees is estimated from fire front length and
        canopy conditions.

        Args:
            cell (Cell): Cell providing crown status, CFB, and canopy data.

        Side Effects:
            Sets ``self.steady_height`` (feet) and ``self.duration``
            (dimensionless time).
        """
        treespecies = self.treespecies
        tnum = 1

        # DBH = diameter at breast height in cm
        # Convert DBH from cm to inches
        DBH = self.dbh / 2.54

        # compute frontal fire distance
        front_dist = self.calc_front_dist(cell)

        if cell._crown_status == CrownStatus.ACTIVE:
            # If the fire is active, use the active crown fire formula
            part1, part2 = self.A[treespecies][:2]
            tnum = max(1, int(front_dist / 5.0))
            self.steady_height = part1 * DBH**part2 * tnum**0.4
        else:
            if cell.cfb > 0.5 and cell.canopy_cover > 50:
                tnum += 1
            if cell.cfb > 0.8:
                tnum += 1
                if cell.canopy_cover > 50:
                    tnum = 6
                if cell.canopy_cover > 80:
                    tnum = 10

            part1, part2 = self.A[treespecies][:2]
            self.steady_height = part1 * DBH**part2 * tnum**0.4

        part3, part4 = self.A[treespecies][2:]
        self.duration = part3 * DBH**(-part4) * tnum**(-0.2)


    def vert_wind_speed(self, height_above_ground: float, canopy_ht: float,
                        wind_speed: float, wind_adj_factor: float) -> float:
        """Compute wind speed at a given height above ground.

        Below the canopy, return the midflame wind speed. Above the canopy,
        use the logarithmic wind profile (Albini & Baughman 1979).

        Args:
            height_above_ground (float): Height above ground (feet).
            canopy_ht (float): Canopy height (feet).
            wind_speed (float): 20-ft or reference wind speed (m/s).
            wind_adj_factor (float): Wind adjustment factor for sub-canopy.

        Returns:
            float: Wind speed at the specified height (ft/s).
        """
        if height_above_ground < canopy_ht:
            # Compute midflame wind speed
            midflame_wind  = wind_adj_factor * wind_speed
            u = m_to_ft(midflame_wind) # midflame wind speed in ft/s
        else:
            # Albini & Baughman 1979 Res. Pap. INT-221
            u = m_to_ft(wind_speed)/(np.log((20 + 0.36 * canopy_ht)/(0.1313 * canopy_ht)))

        return u

    def flight(self, end_curr_time_step: float) -> Set[Cell]:
        """Transport all airborne embers and ignite spot fires.

        Advance each ember through the wind field using a ballistic
        trajectory model until it either lands, exits the domain, or
        reaches the end of the current time step. Embers that land on
        fuel cells ignite spot fires.

        Args:
            end_curr_time_step (float): End time of the current simulation
                step (minutes).

        Returns:
            Set[Cell]: Set of cells where spot fires were ignited.

        Side Effects:
            Updates ``self.embers`` to retain only still-airborne embers.
            Sets ignited cells to ``CellStates.FIRE``.
        """
        spots = set()

        sstep = 0.25 # minutes

        carry = []
        for ember in self.embers:
            sx, sy = ember['x'], ember['y']

            diameter, Z, Zelev, curr_time, elapsed = ember['diam'], ember['height'], ember['start_elev'], ember['curr_time'], ember['elapsed']

            temp_ember = ember

            if curr_time == end_curr_time_step:
                carry.append(ember)
                continue
            else:
                curr_cell = self.get_cell_from_xy(sx, sy, oob_ok=True)                
                wind_speed, wind_dir = curr_cell.curr_wind()

                eH = m_to_ft(curr_cell.canopy_height)

                if eH <= 0.0:
                    eH = 1.0

                zfuel1 = curr_cell.elevation_m
                zfuel2 = curr_cell.elevation_m

                if elapsed == 0:
                    Zelev = zfuel1

                MAXZ = 39000*diameter

                voo = np.sqrt((1910.087*diameter)/0.18) # terminal velocity of particle in ft/s
                tao = (4.8 * voo) / (0.0064 * np.pi * 32)
                Xtot = 0 # reset total horizontal distance traveled
                Z = m_to_ft(Z) # convert to height in feet above original land
                eZelev = m_to_ft(Zelev)
                
                zt1 = 0.0
                rwinddir = wind_dir # degrees

                mZt = np.inf

                abort_flight = False
                while mZt > 1.0 and not abort_flight:
                    # Compute test vertical drop over ember over sstep
                    ztest1 = voo * tao * ((elapsed * 60)/tao - 0.5 * ((elapsed * 60)/tao) ** 2)
                    ztest2 = voo * tao * (((sstep + elapsed) * 60)/tao - 0.5 * (((sstep + elapsed) * 60)/tao) ** 2)

                    if ztest1 < ztest2: # Particle burned out before contacting ground
                        DZRate = (ztest2 - ztest1)/sstep

                    else:
                        tx = sx
                        ty = sy
                        break
                    
                    passA = 0
                    while passA < 3:
                        last_cell = curr_cell

                        HtLimit = 50.0

                        passB = 0
                        while passB < 2:
                            subsstep = HtLimit/DZRate # Sub time step in minutes

                            if (curr_time + subsstep > end_curr_time_step):
                                subsstep = end_curr_time_step - curr_time
                                if subsstep < 0:
                                    subsstep = 0
                                    curr_time = end_curr_time_step
                                HtLimit = subsstep * DZRate # Recalc height limit in terms of new time step
                            
                            zt2 = zt1 + HtLimit
                            if zt1 < 0:
                                Z1HtFromStart = eZelev-(Z-zt1) # Ember dropped below original land height
                            else:
                                Z1HtFromStart = eZelev+(Z-zt1)

                            if zt2 < 0:
                                Z2HtFromStart = eZelev-(Z-zt2) # Ember dropped below original land height
                            else:
                                Z2HtFromStart = eZelev+(Z-zt2)

                            Z1AboveGround = Z1HtFromStart - m_to_ft(zfuel1)
                            Z2AboveGround = Z2HtFromStart - m_to_ft(zfuel2)

                            mZt = ft_to_m(Z2AboveGround)

                            if mZt > 1.0:
                                wind_speed = wind_speed
                                wind_adj_factor = curr_cell.wind_adj_factor

                                UH = self.vert_wind_speed(Z1AboveGround, eH, wind_speed, wind_adj_factor)/2.0 # half of UeH in ft/s
                                UH += self.vert_wind_speed(Z2AboveGround, eH, wind_speed, wind_adj_factor)/2.0 # average UH in ft/s

                                if Z1AboveGround > 0:
                                    if Z2AboveGround > 0:
                                        ZAvgAboveGround = (Z1AboveGround + Z2AboveGround) / 2
                                    else:
                                        ZAvgAboveGround = Z1AboveGround / 2
                                    
                                    if ZAvgAboveGround > 0.1313 * eH:
                                        dxdt = UH/2.03*np.log(ZAvgAboveGround/(0.1313*eH)) # dxdt in ft/s
                                        Xt = dxdt * 60 * subsstep # ft (Albini 1979)
                                        mXt = ft_to_m(Xt) # incremental distance traveled m
                                    
                                    else:
                                        mXt = 0

                                    if passB < 1 and mXt > curr_cell.cell_size - 0.5: # Make sure we don't skip whole cells
                                        # Reduce the sub time step to avoid jumping a whole cell
                                        HtLimit /= 2
                                    else:
                                        break
                                
                                passB += 1

                            else:
                                if mZt < -1.0:
                                    HtLimit = Z1AboveGround + (m_to_ft(zfuel1 - zfuel2))

                                    if HtLimit < 0: # Could happen if ember at zt1 is way below zfuel2 at zt2
                                        mXt = 0.0

                                        passA = 10 if elapsed == 0.0 else 6
                                        break

                                else:
                                    passA = 6
                                    break

                        if passA > 7:
                            break
                        
                        # Update the position of the ember based on the wind direction
                        tx = sx + mXt * np.sin(np.deg2rad(rwinddir))
                        ty = sy + mXt * np.cos(np.deg2rad(rwinddir))

                        curr_cell = self.get_cell_from_xy(tx, ty, oob_ok = True)

                        # Make sure the ember doesn't land outside the sim region
                        if curr_cell is None:
                            # Ember is out of the map
                            passA = 10
                            abort_flight = True
                            break
                        

                        if curr_cell != last_cell:
                            eH = m_to_ft(curr_cell.canopy_height)

                            if eH <= 0.0:
                                eH = 1.0

                            zfuel1 = zfuel2
                            zfuel2 = curr_cell.elevation_m
                            rwinddir = wind_dir

                        else:
                            if zt2 > MAXZ:
                                passA = 10
                                break
                            else:
                                passA +=2
                                break

                        passA += 1

                    if abort_flight:
                        break

                    sx = tx
                    sy = ty
                    elapsed += subsstep
                    Xtot += Xt
                    zt1 = zt2

                    if passA == 10:
                        # Cause failure of ember
                        curr_time = -1
                        mZt = 2.0
                        abort_flight = True
                        break
                    else:
                        if passA == 9:
                            mZt = 0.0
                        if curr_time < end_curr_time_step:
                            curr_time += subsstep
                        else:
                            break

                if np.abs(mZt) < 1.0 and curr_time < end_curr_time_step and not abort_flight:
                    if curr_cell.fuel.burnable:
                        prev_x = temp_ember['x']
                        prev_y = temp_ember['y']

                        dist = np.sqrt((prev_x - tx) ** 2 + (prev_y - ty) ** 2)

                        if dist > self.min_spot_dist_m:
                            
                            if curr_cell.state == CellStates.FUEL:
                                curr_cell.spot_source_xy = (temp_ember['source_x'], temp_ember['source_y'])
                                spots.add(curr_cell) # Add to a set that will handle igniting spot fires
                                curr_cell.get_ign_params(0) # Get ignition parameters for the cell
                                curr_cell._set_state(CellStates.FIRE)
                else:
                    if curr_time == end_curr_time_step:
                        ember['x'] = sx
                        ember['y'] = sy
                        ember['curr_time'] = curr_time
                        ember['elapsed'] = elapsed

                        carry.append(ember)

        self.embers = carry

        return spots
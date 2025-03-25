
from embrs.fire_simulator.cell import Cell
from embrs.utilities.fuel_models import Anderson13
from embrs.utilities.burnup import Burnup

new_ignitions = []


wind_speed = 0
t_amb = 24
dt = 5

# for i in range(13):
cell = Cell(0, 0, 0, 30)
cell._set_cell_data(Anderson13(8), 0, 0, 0, 0, 0)
cell.reaction_intensity = 300
new_ignitions.append(cell)


r_0 = 1.8
dr = 0.4



for cell in new_ignitions:
    # Reset cell burn history
    cell.burn_history = []

    # TODO: implement duff loading in cells
    wdf = 0.74 #cell.wdf

    f_i = cell.reaction_intensity * 0.0031524811 # kW/m2 # TODO: double check this conversion

    u = 0 #wind_speed * cell.wind_adj_factor
    depth = cell.fuel.fuel_depth_ft * 0.3048

    if 2 in cell.fuel.rel_indices:
        mx = cell.fmois[2]
    elif 1 in cell.fuel.rel_indices:
        mx = cell.fmois[1]
    else:
        mx = cell.fmois[0]

    dfm = -0.347 + 6.42 * mx
    dfm = max(dfm, 0.10)

    burn_mgr = Burnup(cell)
    burn_mgr.set_fire_data(5000, f_i, cell.t_r, u, depth, t_amb, r_0, dr, dt, wdf, dfm)

    if burn_mgr.start_loop():
        cont = True
        i = 0
        while cont:
            cont = burn_mgr.burn_loop()
            remaining_fracs = burn_mgr.get_updated_fuel_loading()
            
            # TODO: need to make sure this works as intended
            entry = [0] * len(cell.fuel.w_0)
            j = 0
            for i in range(len(cell.fuel.w_0)):
                if i in cell.fuel.rel_indices:
                    entry[i] = remaining_fracs[j] * cell.fuel.w_n[i] # TODO: when burn is done and cell's actual loading changes, we need to recompute w_n_dead and w_n_live
                    j += 1

            cell.burn_history.append(entry)

    else:
        remaining_fracs = burn_mgr.get_updated_fuel_loading()

        entry = [0] * len(cell.fuel.w_0)
        j = 0
        for i in range(len(cell.fuel.w_0)):
            if i in cell.fuel.rel_indices:
                entry[i] = remaining_fracs[j] * cell.fuel.w_n[i]
                j += 1

        cell.burn_history.append(entry)




print("Done")









# cell = Cell(0, 0, 0, 30, 0, 0, 0, 0, 0, fuel_type=Anderson13(10))

# burn_mgr = Burnup(cell)

# # Store in cell
# reaction_intensity = 300 # Need in kW/m2, can be calculated from Rotermel

# # Weather stream
# wind_speed = 0.0 # Windspeed at top of fuelbed (m/s) 

# # In cell
# depth = cell.fuel.fuel_depth_ft * 0.3048

# # Weather stream
# t_amb = 27 # ambient temperature (C)


# # In FARSITE the parameters r0 and dr serve as geometric constants used in the
# # burn‐calculation model. In the numerical integration (for example, within 
# # BurnUp::Step, where they are used to compute a distance r in the heat‐transfer
# #  calculations) r0 represents the baseline or “starting” effective contact 
# # distance and dr is an incremental adjustment (or effective “width” increment)
# # applied to that baseline. They are preset constants (1.8 and 0.4 in this 
# # implementation) because FARSITE’s underlying combustion model was calibrated
# # using fixed values for these geometric effects rather than computing them
# # dynamically from fuel properties.
# r_0 = 1.8
# dr = 0.4

# time_step = 5 # sim time-step in our case

# # Store in cell
# wdf = 0.74 # duff loading kg/m2 (from fccs data product)



# if 2 in cell.fuel.rel_indices:
#     mx = cell.fmois[2]
# elif 1 in cell.fuel.rel_indices:
#     mx = cell.fmois[1]
# else:
#     mx = cell.fmois[0]

# dfm = -0.347 + 6.42 * mx

# dfm = max(dfm, 0.10)



# for i in range(100):

#     burn_mgr.set_fire_data(5000, reaction_intensity, cell.t_r, wind_speed, depth, t_amb, r_0, dr,
#                         time_step, wdf, dfm)


#     if (burn_mgr.start_loop()):

#         cont = True
#         i = 0
#         while cont:
#             cont = burn_mgr.burn_loop()
#             burn_mgr.get_updated_fuel_loading()

#         print(f"Done with Burnout")

#     else:
#         print(f"Start loop returned false")
#         burn_mgr.get_updated_fuel_loading()

# print("Done")
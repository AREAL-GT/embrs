# Creating a Map

EMBRS comes equipped with a map generation tool to guide you through generating maps that can be used by the simulation.

## **Step 1: Open Map Generator Tool**

- Run the following command in your terminal:

```
python -m embrs.map_generator
```

- You should now see a window like this:
```{figure} /images/mapgen_opening_window.png
---
scale: 40%
---
```

## **Step 2: Choose Map Save Destination and Import Raw Data**

```{note}
If you do not have raw data yet, you can use the provided sample data in the 'sample_raw_data' folder.
```

### Map Save Destination
- The first field in the map generator window prompts you for a folder where the resulting map files will be saved to.
- Click 'Browse' to navigate to your desired folder and select it.
```{warning}
Each map you make must have its own dedicated folder, so if you select a folder that already has map files in it, these will be overwritten.
```

### Fuel Map
- The next field asks for a fuel map as input. For this step there are two options:
    - Option 1: **Select a LANDFIRE fuel map**
        - Click 'Browse' and navigate to your LANDFIRE data.
        - Open the folder called 'LF2022_FBFM13_230_CONUS' and select the file ending in '.tif'.

        ```{figure} /images/mapgen_fuel_file.png
        ---
        scale: 70%
        ---
        ```

        - This will load the fuel map from the region where the data was pulled from.

    - Option 2: **Select a uniform fuel map**
        - Check the box next to 'Uniform Fuel'.
        - Select the fuel type you would like your map to be from the options labeled 'Uniform Fuel Type'.

        ```{figure} /images/mapgen_fuel_options.png
        ---
        scale: 100%
        ---
        ```

        - This will create a fuel map that consists entirely of the fuel type you have selected.

### Elevation Map
- The next field asks for an elevation map as input. For this step there are two options:
    - Option 1: **Select a LANDFIRE elevation map** 
        - Click 'Browse' and navigate to your LANDFIRE data.
        - Open the folder called 'LF2020_Elev_220_CONUS' and select the file ending in '.tif'.

        ```{figure} /images/mapgen_elev_file.png
        ---
        scale: 70%
        ---
        ```

        - This will load the elevation map from the region where the data was pulled from.

    - Option 2: **Select a uniform elevation map**
        - Check the box next to 'Uniform Elevation'.
        - This will create a completely flat elevation map.

### Roads
- If you would like to import the roads for the region of interest check the box next to 'Import Roads from OpenStreetMap'.
- With this option on the map generator will use the OpenStreetMap API to get the real road data for the region.
- For more information on how roads are modeled in EMBRS see [Roads](fire_modelling:roads).

### Width and Height Fields
- Finally, there are fields for entering the width and height in meters of the EMBRS map.
- These options are only available to be specified if you are using both a uniform fuel map and uniform elevation.
    - If you are in this situation, simply type in the width and height in meters to define the dimensions of your map.

    ```{figure} /images/mapgen_all_uniform.png
    ---
    scale: 40%
    ---
    ```

- Otherwise, the width and height of the map will be determined by the input files.

**Once you are finished inputting all your selections press 'Submit'**

## **Step 3: Interactive Map**
- After hitting submit the tool should load an interactive visualization of your map like below:

```{figure} /images/mapgen_interactive_map.png
---
scale: 25%
---
```

- You can pan around the map by holding down right click and moving your mouse.

```{figure} /images/mapgen_pan.gif
---
scale: 150%
---
```

- You can zoom in and out of the map using the scroll wheel on your mouse.

```{figure} /images/mapgen_zoom.gif
---
scale: 150%
---
```

- Press 'Reset View' in the bottom left at any time to get the original view.

```{figure} /images/mapgen_reset.gif
---
scale: 150%
---
```

## **Step 4: Draw Initial Ignition Region(s)**
- It is now time to specify the initial ignition region, this is the region(s) that starts on fire upon the simulation starting

- Start by clicking on the map where you want an ignition region
- Now, when you move the mouse you will see a red dashed line from the initial point you clicked to your cursor, this serves as a preview for the next line segment you draw
- Click as many points as you need to define your region and then complete the polygon by clicking on the point where you started (the line should snap to the initial point)
- Now, in the bottom right you should see options to 'Accept' or 'Decline' the polygon you just drew. Click on the appropriate option

**Drawing first ignition region and accepting:**

```{figure} /images/mapgen_ignition1.gif
---
scale: 150%
---
```

**Drawing second ignition region and declining:**

```{figure} /images/mapgen_ignition2.gif
---
scale: 150%
---
```

- Once you have accepted a ignition region you can either draw another polygon following the same steps above or apply and save the drawn polygons
- If you would like to save the drawn polygons click 'Apply' in the top right, this will advance you to the next step
- If you would like to erase all the polygons you have drawn, you can press the 'Clear' button in the top right

(map_creation:fire_breaks)=
## **Step 5: Draw Fire-breaks (optional)**
- After drawing the initial ignition regions you will now be prompted to draw fire-breaks on the map
- If you do not wish to specify any fire-breaks you can skip this step by pressing 'No Fire Breaks'. This will conclude the map generation process
- Fire-breaks are specified as a series of line segments. Left-click on the map where you would like to start the fire-break and draw as many line segments as you would like
- Once you have completed at least one line segment you will have the 'Accept' and 'Decline' options in the bottom right which you can use at any time.
    - Pressing 'Accept' will save the completed line segments and prompt you to enter a percent fuel remaining along the fire-break. Enter a value between 0-100 which specifies
    what percentage of the original fuel is remaining along the fire-break

    ```{figure} /images/mapgen_firebreak1.gif
    ---
    scale: 150%
    ---
    ```
    
    
    - Pressing 'Decline' will delete the current set of line segments

    ```{figure} /images/mapgen_firebreak2.gif
    ---
    scale: 150%
    ---
    ```

- Once you have accepted a fire-break you can either draw another one following the same steps above or you can apply and save the drawn fire-breaks
- If you would like to save the drawn fire-breaks click 'Apply' in the top right, this will conclude the map generation process
- If you would like to erase all the fire-breaks you have drawn, you can press the 'Clear' button in the top right

(map_creation:files)=
## **Step 6: Resulting Files**
- The map generator tool window will close automatically once you are finished
- You should now be able to navigate to the location you selected as your map's save destination and see the following files:
    - "mapFolderName".json 
    - fuel.npy
    - topography.npy
    - roads.pkl (if roads imported)

```{note}
If a user wanted to manually define a fuel and topography map, this could be done by creating numpy arrays and saving them in .npy files, but this
method is not directly supported. 
```

- You must keep all of these files in the same directory for it to be used by EMBRS
- This folder is what will be loaded when [running a sim](./running_sim.md)
- You can view information about the map by opening the .json file
    - This file contains all the information the simulation needs to load the map
    - In this file you can make manual adjustments to the initial igntion and fire breaks
    - See the sample .json file below

```{note}
If you change the name of the map folder or move it you will need to change the file paths in the resulting .json file.
```

**You now have everything you need to run an EMBRS simulation**

**Sample map .json file**
```
{
    "geo_bounds": {
        "south": 48.107,
        "north": 48.2066,
        "west": -120.5224,
        "east": -120.3553
    },
    "fuel": {
        "file": "/path/to/folder/fuel.npy",
        "width_m": 12270,
        "height_m": 10680,
        "rows": 356,
        "cols": 409,
        "resolution": 30,
        "uniform": false
    },
    "topography": {
        "file": "/path/to/folder/tutorial_map/topography.npy",
        "width_m": 12270,
        "height_m": 10680,
        "rows": 10740,
        "cols": 12300,
        "resolution": 1.0,
        "uniform": false
    },
    "roads": {
        "file": "/path/to/folder/tutorial_map/roads.pkl"
    },
    "initial_ignition": [
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [
                        3851.669495402206,
                        5201.817838667499
                    ],
                    [
                        3777.4476936999454,
                        4756.487028453935
                    ],
                    [
                        4000.113098806727,
                        4526.399443176927
                    ],
                    [
                        4356.377746977578,
                        4615.465605219641
                    ],
                    [
                        4682.953674467524,
                        5053.374235262978
                    ],
                    [
                        4638.420593446168,
                        5535.815946327672
                    ],
                    [
                        4297.000305615769,
                        5669.4151893917415
                    ],
                    [
                        3851.669495402206,
                        5201.817838667499
                    ]
                ]
            ]
        }
    ],
    "fire_breaks": [
        {
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [
                        4864.959266243655,
                        6938.026285871776
                    ],
                    [
                        5228.646094584733,
                        6492.695475658213
                    ],
                    [
                        5495.844580712871,
                        5802.432719827188
                    ],
                    [
                        5644.288184117392,
                        5297.724468251815
                    ],
                    [
                        5629.44382377694,
                        4770.749676165765
                    ]
                ]
            },
            "fuel_value": 10.0
        },
        {
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [
                        2772.413026025168,
                        4806.718451932299
                    ],
                    [
                        3049.022836609153,
                        4184.346378118333
                    ],
                    [
                        3819.5787375216833,
                        3720.0370532095008
                    ],
                    [
                        4827.228761791915,
                        3561.974304304367
                    ]
                ]
            },
            "fuel_value": 15.0
        }
    ]
}
```

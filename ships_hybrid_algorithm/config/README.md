# CONFIG

Config file for Ship Movement Model

## Parameters

* simulation_steps -> Number of steps (seconds) of the simulation
* max_speed_range -> Max speed of the ship in this range (random)
* speed_variation -> if enabled, the maximum delta between targeted speed and real speed
* directional_variation -> if enabled, ship heading can be deviated
* resolution -> increase for less compute time, but ships can be stuck
* obstacle_threshold -> how close a ship can approach an obstacle

* ports [x,y] -> coordinates of the port, in grid coordinates
* speed_limit_zone[x,y] -> coordinates and max speed of a speed limit zone, in grid coordinates
* obstacle [[x1,y1],...,[xn,yn]] -> coordinates of an obstacle, in grid coordinates

* geospatial_bounds -> min and max lat/lon of the map, in decimal degrees


## How to have accurate obstacle coordinates

1. Install and open qgis (https://qgis.org/)
2. Create project. Specify EPSG 4326 (WGS 84)
3. Install QuickOSM plugin
4. Using QuickOSM quick request, request KEY=boundaries and VALUE=administrative. You can filter to have only relevant boundaries
5. Now you will have polygons' coordinates with your obstacle. Convert them in grid coordinates and use them in configV2.json

## My map is distorded

You need x,y to have the same ratio aspect as the lon,lat of your map. For exemple :

1. Get min/max lon/lat of your area (in degrees)
2. Decide height of the map in grid coordinates (eg, 500)
3. Get the "mean latitude" of your map (eg, $\phi_0$).
4. Ratio is :
$R \approx \cos(\phi_0)\cdot \frac{\Delta lon}{\Delta lat}$
5. Now you have : $x = y \times R$

## How to convert lon,lat to x,y

You have defined width and height of the map. You also have min and max lon,lat of the map in decimal degrees.
$x=\frac{\mathrm{lon}-\mathrm{min\_lon}}{\mathrm{max\_lon}-\mathrm{min\_lon}}\cdot \mathrm{width},\ y=\frac{\mathrm{lat}-\mathrm{min\_lat}}{\mathrm{max\_lat}-\mathrm{min\_lat}}\cdot \mathrm{height}$

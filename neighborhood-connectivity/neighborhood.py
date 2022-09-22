# base neighborhood class

import os
import json
import time
from typing import Optional, Sequence, Union, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np
import overpass
import osm2geojson
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint

# TODO - move to dedicated file
def run_overpass_query(
    query: str,
    retries: int = 3,
    op: overpass.API = None,
    cache_path: str = None,
    verbosity: str = "body"
) -> gpd.GeoDataFrame:
    """Read a query from Overpass API, or maybe from a file"""
    if op is None:
        op = overpass.API(timeout=600)
    if os.path.isfile(cache_path):
        return gpd.read_file(cache_path)
    while retries:
        try:
            xml = op.get(query, responseformat="xml", verbosity=verbosity)
            print(f"Succesfully executed query..")
            break
        except Exception as e:
            print(f"Failed query due to {e}")
            retries -= 1
            if retries > 0:
                print(f"Failed on try {retries}, pausing 5 seconds before continuing")
                time.sleep(5)
                continue
            else:
                raise
    with open(cache_path, 'w') as f:
        json.dump(
            osm2geojson.xml2geojson(xml),
            f
        )
    gdf = gpd.read_file(cache_path)
    if not cache_path:
        os.remove(cache_path)
    else:
        print(f"Succesfully cached data {os.path.basename(cache_path)}")
    return gdf

# TODO - move to dedicated file
QUERY_PATH = os.path.join(
    os.path.dirname(__file__),
    "queries"
)
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data"
)
TRIMET_FREQUENCY = {
    "Orange":4,
    "Yellow":4,
    "Red":4,
    "Green":4,
    "Blue":4,
    "2":5, # FX is not in OSM yet
    "4":4,
    "6":4,
    "8":4,
    "9":4,
    "12":4,
    "14":4,
    "15":4,
    "20":4,
    "33":4,
    "54":4,
    "55":4,
    "56":4,
    "57":4,
    "72":4,
    "73":4,
    "75":4,
    "76":4
}

class Neighborhood(object):
    """abstract neighborhood idea 
        - has a name, id, adjacent neighborhoods
        geometry
    """
    def __init__(
        self,
        name: str,
        other: Optional[str] = None,
        state: str = "Oregon",
        city: str = "Portland"
    ) -> None:
        self.city = city
        self.state = state
        self.name = name
        self.nhood, self.others = self._fetch_data()
        self.other_names = set(self.others["name"].unique())
        if other is not None:
            if other not in self.other_names:
                raise NotImplementedError("No support for non-adjacent neighborhoods")
            self.other = other
        else:
            self.other = None
        with open(os.path.join(DATA_PATH, f"{state} {city}","neighborhood_populations.json")) as f:
            self._nhood_pops = json.load(f)
        self.population = self._nhood_pops.get(name)
        self.other_population = self._nhood_pops.get(other)

    def _fetch_data(
        self,
        cache_path: os.PathLike = None
    ) -> Tuple[gpd.GeoSeries, gpd.GeoDataFrame]:
        # run query for adjacent neighborhoods
        if not cache_path:
            cache_path = os.path.join(
                DATA_PATH,
                f"{self.state} {self.city}",
                f"base_data_{self.name}.geojson"
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        query = os.path.join(
            QUERY_PATH,
            "adjacent_neighborhoods.op"
        )
        with open(query, "r") as f:
            query = f.read()
            query = query.format(self.name)
        gdf = run_overpass_query(
            query=query, 
            cache_path=cache_path, 
            verbosity="geom"
        )
        gdf["name"] = gdf["tags"].apply(lambda x: x.get("name"))
        nhood = gdf[gdf["name"] == self.name].iloc[0]
        other = gdf[gdf["name"] != self.name]
        return nhood, other

    @staticmethod
    def closest_point_on_line(
        point: Point,
        line: Union[LineString, MultiLineString],
        zero_distance_allowed = False
    ) -> Point:
        """Point on a line (or multi-line) closest to another point (either on, or not on line)"""
        min_dist = None
        min_point = None
        if isinstance(line, MultiLineString):
            for line_str in line:
                for pt in line_str.coords:
                    pt = Point(pt)
                    dist = pt.distance(point)
                    if dist <= 0 and not zero_distance_allowed:
                        continue
                    if not min_dist:
                        min_dist = dist
                        min_point = pt
                    elif dist < min_dist:
                        min_dist = dist
                        min_point = pt
        else:
            for pt in line.coords:
                pt = Point(pt)
                dist = pt.distance(point)
                if dist <= 0 and not zero_distance_allowed:
                    continue
                if not min_dist:
                    min_dist = dist
                    min_point = pt
                elif dist < min_dist:
                    min_dist = dist
                    min_point = pt
        if not min_point:
            raise Exception("No closest point found")
        return min_point

    @staticmethod
    def angle_of_three_points(
        pt1: Point,
        pt2: Point,
        pt3: Point
    ) -> float:
        """Calculate angle between three points, with pt2 shared between the vectors"""
        pt1 = np.array(pt1.coords[0])
        pt2 = np.array(pt2.coords[0])
        pt3 = np.array(pt3.coords[0])

        v21 = pt2 - pt1
        v23 = pt2 - pt3
        cosine = np.dot(v21, v23) / (np.linalg.norm(v21) * np.linalg.norm(v23))
        return np.degrees(np.arccos(cosine))

    def fetch_boundary(
        self,
        other: Optional[str] = None
    ) -> Union[LineString, MultiLineString]:
        """Fetch the bonudary between self neighborhood and one other"""
        if not other and self.other:
            other = self.other
        elif not other:
            raise ValueError("Must pass other neighborhood name at class init, or function call")
        if other not in self.other_names:
            raise AttributeError("Non-adjacent neighborhoods have no boundary")
        
        cache_path = os.path.join(
            DATA_PATH,
            f"boundary_data_{'_'.join(sorted((self.name, other)))}.geojson"
        )

        query = os.path.join(
            QUERY_PATH,
            "neighborhood_boundary.op"
        )
        with open(query, "r") as f:
            query = f.read()
            query = query.format(self.name, other)
        gdf = run_overpass_query(
            query=query,
            cache_path=cache_path,
            verbosity="geom"
        )
        geom = gdf["geometry"]
        if len(geom.index) == 1:
            return geom.iloc[0]
        else:
            # create multiline string from boundary
            return MultiLineString(geom.to_list())

    def fetch_routes(
        self,
        other: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """Fetch routes between self neighborhood and one other"""
        if not other and self.other:
            other = self.other
        elif not other:
            raise ValueError("Must pass other neighborhood name at class init, or function call")
        if other not in self.other_names:
            raise NotImplementedError("Have not implemented non-adjacent routes")

        cache_path = os.path.join(
            DATA_PATH,
            f"route_data_{'_'.join(sorted((self.name, other)))}.geojson"
        )

        query = os.path.join(
            QUERY_PATH,
            "routes_between_neighborhoods.op"
        )
        with open(query, "r") as f:
            query = f.read()
            query = query.format(self.name, other)
        gdf = run_overpass_query(
            query=query,
            cache_path=cache_path,
            verbosity="geom"
        )
        # angle of intersection with the boudary..
        boundary = self.fetch_boundary(other=other)
        # TODO - make this have fewer function calls
        
        """ the lambda function is essentially this loop \|/
        print(other)
        for idx in gdf.index:
            print(gdf.iloc[idx])
            geom = gdf.loc[idx]["geometry"]
            pt = boundary.intersection(geom)
            if isinstance(pt, MultiPoint):
                pt = pt.geoms[0]
            # closest point on boundary to pt
            boundary_closest = self.closest_point_on_line(pt, boundary)
            route_closest = self.closest_point_on_line(pt, geom)
            # calculate angle
            angle = self.angle_of_three_points(boundary_closest, pt, route_closest)
            print(angle)"""
        # can 
        gdf["intersection_point"] = gdf["geometry"].apply(
            lambda x: boundary.intersection(x) if isinstance(boundary.intersection(x), Point) else boundary.intersection(x)[0]
        )
        gdf["intersection_angle"] = gdf.apply(
            lambda x: self.angle_of_three_points(
                self.closest_point_on_line(x["intersection_point"], boundary),
                x["intersection_point"],
                self.closest_point_on_line(x["intersection_point"], x["geometry"])
            ),
            axis=1
        )
        print(gdf["intersection_angle"])

        # grade routes for car, ped, and bike
        gdf["highway"] = gdf["tags"].apply(lambda x: x.get("highway"))
        gdf["bicycle"] = gdf["tags"].apply(lambda x: x.get("bicycle"))
        gdf["cycleway"] = gdf["tags"].apply(lambda x: x.get("cycleway"))
        gdf["sidewalk"] = gdf["tags"].apply(lambda x: x.get("sidewalk"))
        gdf["foot"] = gdf["tags"].apply(lambda x: x.get("foot"))
        gdf["maxspeed"] = gdf["tags"].apply(lambda x: x.get("maxspeed","0 mph"))
        gdf["maxspeed"] = gdf["maxspeed"].apply(
            lambda x: x.split(" ")[0] * 1 if x.split(" ")[1] == "mph" else (1/1.6)
        )

        # dicts to determine scoring - TODO: move these to a JSON
        highway_score = {
            "car":{
                "motorway":1,"trunk":1,"primary":0.9,
                "secondary":0.8,"tertiary":0.7,"residential":0.6,
                "unclassified":0.5,"motorway_link":1,"trunk_link":1,
                "primary_link":0.9,"secondary_link":0.8,"tertiary_link":0.7,
            },
            "bike":{
                "primary":0.5,"secondary":0.6,"tertiary":0.7,
                "unclassified":0.7,"residential":0.8,
                "primary_link":0.4,"secondary_link":0.5,
                "tertiary_link":0.6,
                "footway":0.9,"path":0.9,"living_street":0.9,
                "cycleway":1,
            },
            "ped":{
                "primary":0.4,"secondary":0.5,"tertiary":0.6,
                "unclassified":0.7,"residential":0.9,
                "primary_link":0.3,"secondary_link":0.4,
                "tertiary_link":0.5,
                "footway":1,"path":1,"living_street":1,
                "pedestrian":1
            },
        }
        # additive
        special_bike = {
            "cycleway":{
                "lane":0.1,"opposite":0.05,"opposite_lane":0.05,
                "track":0.3,"share_busway":0.1,"opposite_share_busway":0.05,
                "shared_lane":0.05
            },
            "bicycle":{
                "yes":0.05,"no":-10,"designated":0.1,
                "dismount":-10
            }
        }
        # additive
        special_ped = {
            "sidewalk":{
                "right":-0.3,"left":-0.3,
                "no":-10
            },
            "foot":{
                "designated":1
            }
        }
        # speed functions: bad for non-car, cars get priority anyways
        car_speed = lambda x: x 
        bike_speed = lambda x: x if x < 30 else x/2
        ped_speed = lambda x: x if x < 30 else x/2
        # if angle of intersection is < 10 degrees, assume it's a bad one
        angle_function = lambda x: 0 if x < 10 else 1
        # TODO - move this to a better function with maybe fewer lambdas
        gdf["car_score"] = (
            gdf["highway"].apply(lambda x: highway_score["car"].get(x,0)).apply(car_speed)
        ) * (gdf["intersection_angle"].apply(angle_function))
        gdf["bike_score"] = (
            gdf["highway"].apply(lambda x: highway_score["bike"].get(x,0)).apply(bike_speed)
            + gdf["cycleway"].apply(lambda x: special_bike["cycleway"].get(x,0))
            + gdf["bicycle"].apply(lambda x: special_bike["bicycle"].get(x,0))
        ).apply(lambda x: max(min(x, 1),0)) * (gdf["intersection_angle"].apply(angle_function))
        gdf["ped_score"] = (
            gdf["highway"].apply(lambda x: highway_score["ped"].get(x,0)).apply(ped_speed)
            + gdf["sidewalk"].apply(lambda x: special_ped["sidewalk"].get(x,0))
            + gdf["foot"].apply(lambda x: special_ped["foot"].get(x,0))
        ).apply(lambda x: max(min(x, 1),0)) * (gdf["intersection_angle"].apply(angle_function))

        return gdf

    def fetch_transit_routes(
        self,
        other: Optional[str] = None
    ) -> set:
        """Transit routes between neighborhoods"""
        if not other and self.other:
            other = self.other
        elif not other:
            raise ValueError("Must pass other neighborhood name at class init, or function call")
        if other not in self.other_names:
            raise NotImplementedError("Have not implemented non-adjacent routes")

        cache_path = os.path.join(
            DATA_PATH,
            f"transit_data_{'_'.join(sorted((self.name, other)))}.geojson"
        )

        query = os.path.join(
            QUERY_PATH,
            "transit_between_neighborhoods.op"
        )
        with open(query, "r") as f:
            query = f.read()
            query = query.format(self.name, other)
        gdf = run_overpass_query(
            query=query,
            cache_path=cache_path,
            verbosity="geom"
        )
        gdf["ref"] = gdf["tags"].apply(lambda x: x.get("ref"))
        gdf["mode"] =  gdf["tags"].apply(lambda x: x.get("route"))
        pd.options.mode.chained_assignment = None 
        df = gdf[["ref","mode"]]
        df["frequency"] = df["ref"].apply(lambda x: TRIMET_FREQUENCY.get(x))
        df.drop_duplicates(inplace=True)
        df["frequency"].fillna(1.5,inplace=True)
        df["capacity"] = np.where(
            df["mode"] == "light_rail",
            200,
            np.where(
                df["mode"].isin(["bus","tram"]),
                50,
                10 #catch-all...
            )
        )
        pd.options.mode.chained_assignment = "warn"
        return df

    def score(
        self,
        other: Optional[str] = None
    ) -> dict:
        """car, bike, ped, transit score between self and other"""
        if not other and self.other:
            other = self.other
        elif not other:
            raise ValueError("Must pass other neighborhood name at class init, or function call")
        if other not in self.other_names:
            raise NotImplementedError("Have not implemented non-adjacent neighborhoods")
        
        if not self.other_population:
            other_population = self._nhood_pops.get(other)
        else:
            other_population = self.other_population

        route_score = self.fetch_routes(other=other)
        transit = self.fetch_transit_routes(other=other)
        
        car_score = (
            route_score["car_score"].mean() 
            + route_score["car_score"].median()
        ) / 2
        bike_score = (
            route_score["bike_score"].mean()
            + route_score["bike_score"].median()
        ) / 2
        ped_score = (
            route_score["ped_score"].mean()
            + route_score["ped_score"].median()
        ) / 2
        transit_score = (
            (transit["frequency"] * transit["capacity"]).sum()
            / (self.population + other_population)
        )
        return {
            "car":car_score,
            "bike":bike_score,
            "ped":ped_score,
            "transit":transit_score
        }
    
    def all_scores(self) -> dict:
        """score all neighbors"""
        return {neighbor:self.score(other=neighbor) for neighbor in self.other_names}

if __name__ == "__main__":
    n = Neighborhood("Brooklyn")
    data = n.all_scores()
    print(data)
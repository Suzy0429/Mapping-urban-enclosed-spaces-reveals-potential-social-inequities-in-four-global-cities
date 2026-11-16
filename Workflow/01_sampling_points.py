# -*- coding: utf-8 -*-
"""Step 1 - street-view sampling points and viewing angles.

For every parcel edge, place sampling points along a 10 m outward buffer at
50 m intervals, and compute for each point the compass bearing from the point
to the parcel's visual centroid.  A street-view image is later requested at
that point with a 90 degree field of view centred on that bearing, so the
parcel boundary is photographed frontally rather than obliquely.

Inputs
  --parcels   polygon feature class of parcel units, any projected CRS
  --out-dir   directory for the outputs

Outputs
  <out-dir>/points_<city>.csv   point_id, parcel_id, lon, lat, angle_deg
  <out-dir>/points_<city>.shp   the same points in the working metric CRS
  <out-dir>/centroids_<city>.shp

Requires ArcGIS Pro (arcpy), which is what produced the published sampling
points.  Run with ArcGIS Pro's own Python, e.g.
  "C:/Program Files/ArcGIS/Pro/bin/Python/envs/arcgispro-py3/python.exe" \
      Workflow/01_sampling_points.py --parcels parcels_beijing.shp \
      --out-dir out --city beijing
"""
import argparse
import csv
import math
import os

import arcpy


def utm_epsg_for(feature_class, scratch):
    """UTM zone of the layer's centre, so the metric step is locally correct."""
    wgs84 = os.path.join(scratch, "parcels_wgs84.shp")
    arcpy.management.Project(feature_class, wgs84, arcpy.SpatialReference(4326))
    ext = arcpy.Describe(wgs84).extent
    lon = (ext.XMin + ext.XMax) / 2.0
    lat = (ext.YMin + ext.YMax) / 2.0
    zone = int((lon + 180.0) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def prepare_parcels(parcels, scratch, epsg):
    """Project to the working metric CRS and carry the parcel id across."""
    projected = os.path.join(scratch, "parcels_projected.shp")
    if arcpy.Exists(projected):
        arcpy.management.Delete(projected)
    arcpy.management.Project(parcels, projected, arcpy.SpatialReference(epsg))
    arcpy.management.AddField(projected, "parcel_id", "LONG")
    arcpy.management.CalculateField(projected, "parcel_id", "!OBJECTID!", "PYTHON3")
    return projected


def build_buffers(projected, scratch, distance):
    """One buffer polygon per parcel, outside the parcel only."""
    buffers = os.path.join(scratch, "buffers.shp")
    if arcpy.Exists(buffers):
        arcpy.management.Delete(buffers)
    arcpy.analysis.Buffer(
        in_features=projected,
        out_feature_class=buffers,
        buffer_distance_or_field="%d Meters" % distance,
        line_side="OUTSIDE_ONLY",
        line_end_type="ROUND",
        dissolve_option="NONE",
    )
    return buffers


def build_centroids(projected, scratch):
    """Visual centroid: a point guaranteed to fall inside the parcel."""
    centroids = os.path.join(scratch, "centroids.shp")
    if arcpy.Exists(centroids):
        arcpy.management.Delete(centroids)
    arcpy.FeatureToPoint_management(projected, centroids, "INSIDE")
    return centroids


def ring_lines(polygon, spatial_reference):
    """Every ring of a (possibly multi-part) polygon as a closed polyline."""
    rings = []
    for part in polygon:
        points = arcpy.Array([p for p in part if p is not None])
        if len(points) < 3:
            continue
        if not points[0].equals(points[-1]):
            points.add(points[0])
        line = arcpy.Polyline(points, spatial_reference)
        if line.length > 0:
            rings.append(line)
    return rings


def sample_points(buffers, out_fc, spacing, min_points):
    """Points at equal arc-length intervals along every buffer ring."""
    sr = arcpy.Describe(buffers).spatialReference
    arcpy.management.CreateFeatureclass(
        out_path=os.path.dirname(out_fc), out_name=os.path.basename(out_fc),
        geometry_type="POINT", spatial_reference=sr)
    arcpy.management.AddField(out_fc, "point_id", "LONG")
    arcpy.management.AddField(out_fc, "parcel_id", "LONG")

    fields = ["OID@", "SHAPE@", "parcel_id"]
    next_id = 1
    with arcpy.da.SearchCursor(buffers, fields) as src, \
            arcpy.da.InsertCursor(out_fc, ["SHAPE@", "point_id", "parcel_id"]) as dst:
        for _, polygon, parcel_id in src:
            if polygon is None:
                continue
            for line in ring_lines(polygon, sr):
                # equal intervals whose count is the closest integer to L/50
                n = max(min_points, int(round(line.length / spacing)))
                step = line.length / n
                for i in range(n):
                    point = line.positionAlongLine(i * step)
                    if point:
                        dst.insertRow([point, next_id, parcel_id])
                        next_id += 1
    return next_id - 1


def bearing_to_centroid(points_fc, centroids_fc):
    """Clockwise degrees from due north, from each point to its parcel centroid."""
    centroid = {}
    with arcpy.da.SearchCursor(centroids_fc, ["parcel_id", "SHAPE@X", "SHAPE@Y"]) as cur:
        for parcel_id, x, y in cur:
            centroid[parcel_id] = (x, y)

    arcpy.management.AddField(points_fc, "angle_deg", "DOUBLE")
    with arcpy.da.UpdateCursor(
            points_fc, ["parcel_id", "SHAPE@X", "SHAPE@Y", "angle_deg"]) as cur:
        for parcel_id, x, y, _ in cur:
            cx, cy = centroid[parcel_id]
            # atan2(east, north) is already measured clockwise from north
            cur.updateRow([parcel_id, x, y, math.degrees(math.atan2(cx - x, cy - y)) % 360.0])


def write_csv(points_fc, csv_path):
    """Longitude/latitude plus the bearing, which is what the crawler consumes."""
    wgs84 = arcpy.SpatialReference(4326)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["point_id", "parcel_id", "lon", "lat", "angle_deg"])
        fields = ["point_id", "parcel_id", "SHAPE@", "angle_deg"]
        with arcpy.da.SearchCursor(points_fc, fields) as cur:
            for point_id, parcel_id, geometry, angle in cur:
                geographic = geometry.projectAs(wgs84).firstPoint
                writer.writerow([point_id, parcel_id,
                                 round(geographic.X, 6), round(geographic.Y, 6),
                                 round(angle, 2)])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parcels", required=True, help="parcel polygon feature class")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--city", required=True, help="label used in the output file names")
    parser.add_argument("--buffer", type=float, default=10.0,
                        help="outward buffer distance in metres (default 10)")
    parser.add_argument("--spacing", type=float, default=50.0,
                        help="target spacing between sampling points in metres (default 50)")
    parser.add_argument("--min-points", type=int, default=4,
                        help="minimum points per buffer ring (default 4)")
    parser.add_argument("--utm-epsg", type=int, default=None,
                        help="override the working metric CRS")
    args = parser.parse_args()

    arcpy.env.overwriteOutput = True
    scratch = os.path.join(args.out_dir, "scratch_%s" % args.city)
    os.makedirs(scratch, exist_ok=True)

    epsg = args.utm_epsg or utm_epsg_for(args.parcels, scratch)
    print("working metric CRS: EPSG:%d" % epsg)

    projected = prepare_parcels(args.parcels, scratch, epsg)
    buffers = build_buffers(projected, scratch, args.buffer)
    centroids = build_centroids(projected, scratch)

    points_fc = os.path.join(args.out_dir, "points_%s.shp" % args.city)
    if arcpy.Exists(points_fc):
        arcpy.management.Delete(points_fc)
    total = sample_points(buffers, points_fc, args.spacing, args.min_points)
    print("sampling points: %d" % total)

    bearing_to_centroid(points_fc, centroids)

    out_centroids = os.path.join(args.out_dir, "centroids_%s.shp" % args.city)
    if arcpy.Exists(out_centroids):
        arcpy.management.Delete(out_centroids)
    arcpy.management.CopyFeatures(centroids, out_centroids)

    csv_path = os.path.join(args.out_dir, "points_%s.csv" % args.city)
    write_csv(points_fc, csv_path)
    print("wrote %s" % csv_path)


if __name__ == "__main__":
    main()

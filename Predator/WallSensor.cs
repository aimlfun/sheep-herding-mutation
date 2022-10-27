using SheepHerderMutation;
using SheepHerderMutation.Configuration;
using SheepHerderMutation.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace SheepHerderMutation.Predator;

/// <summary>
///  __        __    _ _   ____                            
///  \ \      / /_ _| | | / ___|  ___  _ __ ___  ___  _ __ 
///   \ \ /\ / / _` | | | \___ \ / _ \ '_ \/ __|/ _ \| '__|
///    \ V V  / (_| | | |  ___) |  __/ | | \__ \ (_) | |   
///     \_/\_/ \__,_|_|_| |____/ \___|_| |_|___/\___/|_|   
/// </summary>
internal class WallSensor
{
    /// <summary>
    /// Stores the "triangles" that make up the full sweep of the sensor (with or without wall).
    /// </summary>
    private readonly List<PointF[]> wallSensorSweepTrianglePolygonsInDeviceCoordinates = new();

    /// <summary>
    /// Stores the "triangles" that contain wall.
    /// </summary>
    private readonly List<PointF[]> wallSensorTriangleTargetIsInPolygonsInDeviceCoordinates = new();

    /// <summary>
    /// 
    /// </summary>
    /// <param name="angleLifeFormIsPointing"></param>
    /// <param name="predatorLocation"></param>
    /// <param name="heatSensorRegionsOutput"></param>
    /// <returns></returns>
    internal double[] Read(double angleLifeFormIsPointing, PointF predatorLocation, out double[] heatSensorRegionsOutput)
    {
        wallSensorTriangleTargetIsInPolygonsInDeviceCoordinates.Clear();
        wallSensorSweepTrianglePolygonsInDeviceCoordinates.Clear();

        int SamplePoints = Config.DogWallSensorSamplePoints;

        heatSensorRegionsOutput = new double[SamplePoints];

        // e.g 
        // input to the neural network
        //   _ \ | / _   
        //   0 1 2 3 4 
        //        
        double fieldOfVisionStartInDegrees = 0;

        //   _ \ | / _   
        //   0 1 2 3 4
        //   [-] this
        double sensorVisionAngleInDegrees = 45;

        //   _ \ | / _   
        //   0 1 2 3 4
        //   ^ this
        double sensorAngleToCheckInDegrees = fieldOfVisionStartInDegrees - sensorVisionAngleInDegrees / 2 + angleLifeFormIsPointing;

        // how far from the center of the predator it looks for a wall 
        double DepthOfVisionInPixels = Config.DogSensorWallDepthOfVisionInPixels;

        for (int LIDARangleIndex = 0; LIDARangleIndex < SamplePoints; LIDARangleIndex++)
        {
            //     -45  0  45
            //  -90 _ \ | / _ 90   <-- relative to direction of dog, hence + angle dog is pointing
            double LIDARangleToCheckInRadiansMin = MathUtils.DegreesInRadians(sensorAngleToCheckInDegrees);
            double LIDARangleToCheckInRadiansMax = LIDARangleToCheckInRadiansMin + MathUtils.DegreesInRadians(sensorVisionAngleInDegrees);

            /*  p1        p2
             *   +--------+
             *    \      /
             *     \    /     this is our imaginary "wall sensor" triangle
             *      \  /
             *       \/
             *    predator
             */
            PointF p1 = new((float)(Math.Sin(LIDARangleToCheckInRadiansMin) * DepthOfVisionInPixels + predatorLocation.X),
                            (float)(Math.Cos(LIDARangleToCheckInRadiansMin) * DepthOfVisionInPixels + predatorLocation.Y));

            PointF p2 = new((float)(Math.Sin(LIDARangleToCheckInRadiansMax) * DepthOfVisionInPixels + predatorLocation.X),
                            (float)(Math.Cos(LIDARangleToCheckInRadiansMax) * DepthOfVisionInPixels + predatorLocation.Y));

            wallSensorSweepTrianglePolygonsInDeviceCoordinates.Add(new PointF[] { predatorLocation, p1, p2 });

            heatSensorRegionsOutput[LIDARangleIndex] = 1; // no target in this direction

            // check each "target" rectangle and see if it intersects with the sensor.            
            foreach (PointF[] points in LearnToHerd.s_lines)
            {
                for (int i = 0; i < points.Length - 1; i++) // -1, because we're doing line "i" to "i+1"
                {
                    PointF point1 = points[i];
                    PointF point2 = points[i + 1];
                    PointF intersection2 = new();
                    PointF intersection3 = new();
                    PointF intersection4 = new();

                    bool detectedWallBetweenPredatorAndEdgeOfLeftSideOfTriangle = false;
                    bool detectedWallBetweenPredatorAndEdgeOfRightSideOfTriangle = false;
                    bool detectedWallBetweenFurthestEdgeOfTriangle = false;

                    /*  p1         p2
                     *   +--------+
                     *    \      /
                     *     \   \\ wall    this is our imaginary "wall sensor" triangle
                     *      \  /
                     *       \/
                     *    predator
                     */
                    if (MathUtils.GetLineIntersection(predatorLocation, p1, point1, point2, out intersection2))
                    {
                        detectedWallBetweenPredatorAndEdgeOfLeftSideOfTriangle = true;
                    }

                    /*  p1         p2
                     *   +--------+
                     *    \      /
                     * wall//   /     this is our imaginary "wall sensor" triangle
                     *      \  /
                     *       \/
                     *    predator
                     */
                    if (MathUtils.GetLineIntersection(predatorLocation, p2, point1, point2, out intersection3))
                    {
                        detectedWallBetweenPredatorAndEdgeOfRightSideOfTriangle = true;
                    }


                    /*  p1  wall   p2
                     *   +---||---+
                     *    \      /
                     *     \    /     this is our imaginary "wall sensor" triangle
                     *      \  /
                     *       \/
                     *    predator
                     */
                    if (MathUtils.GetLineIntersection(p1, p2, point1, point2, out intersection4))
                    {
                        detectedWallBetweenFurthestEdgeOfTriangle = true;
                    }

                    if (!detectedWallBetweenPredatorAndEdgeOfLeftSideOfTriangle &&
                        !detectedWallBetweenPredatorAndEdgeOfRightSideOfTriangle &&
                        !detectedWallBetweenFurthestEdgeOfTriangle) continue;

                    if (!detectedWallBetweenPredatorAndEdgeOfRightSideOfTriangle) intersection3 = intersection2;
                    if (!detectedWallBetweenPredatorAndEdgeOfLeftSideOfTriangle) intersection2 = intersection3;

                    if (detectedWallBetweenFurthestEdgeOfTriangle)
                    {
                        intersection2 = intersection4;
                        intersection3 = intersection2;
                    }

                    PointF intersection = new((intersection2.X + intersection3.X) / 2,
                                                     (intersection2.Y + intersection3.Y) / 2
                                                        );

                    double dist = MathUtils.DistanceBetweenTwoPoints(predatorLocation, intersection).Clamp(0F, (float)DepthOfVisionInPixels);

                    double mult = dist / DepthOfVisionInPixels;

                    if (mult < heatSensorRegionsOutput[LIDARangleIndex])
                    {
                        heatSensorRegionsOutput[LIDARangleIndex] = mult;  // closest
                    }
                }
            }

            if (heatSensorRegionsOutput[LIDARangleIndex] != 1)
            {
                wallSensorTriangleTargetIsInPolygonsInDeviceCoordinates.Add(new PointF[] { predatorLocation, p1, p2 });
            }
            heatSensorRegionsOutput[LIDARangleIndex] = 1 - heatSensorRegionsOutput[LIDARangleIndex];

            //   _ \ | / _         _ \ | / _   
            //   0 1 2 3 4         0 1 2 3 4
            //  [-] from this       [-] to this
            sensorAngleToCheckInDegrees += sensorVisionAngleInDegrees;
        }

        return heatSensorRegionsOutput;
    }

    /// <summary>
    /// Draws the full triangle sweep range.
    /// +--------+
    ///  \      /
    ///   \    /     this is our imaginary "sensor" triangle
    ///    \  /
    ///     \/
    /// </summary>
    /// <param name="g"></param>
    /// <param name="triangleSweepColour"></param>
    internal void DrawFullSweepOfHeatSensor(Graphics g, Color triangleSweepColour)
    {
        bool showSegmentNumber = false;

        using SolidBrush brushOrange = new(triangleSweepColour);
        using Pen pen = new(Color.FromArgb(60, 100, 100, 100));

        int i = 0;

        foreach (PointF[] point in wallSensorSweepTrianglePolygonsInDeviceCoordinates)
        {
            g.FillPolygon(brushOrange, point);
            g.DrawPolygon(pen, point);

            if (showSegmentNumber)
            {
                PointF p = CentreOfPoints(point);
                Font f = new("Arial", 7);
                SizeF size = g.MeasureString(i.ToString(), f);
                g.DrawString(i.ToString(), f, Brushes.Black, p.X - size.Width / 2, p.Y - size.Height / 2);
                ++i;
            }
        }
    }

    /// <summary>
    /// Determines the centre of all the points. It's possibly too approximate, 
    /// by taking min/max boundaries and halving.
    /// </summary>
    /// <param name="point"></param>
    /// <returns></returns>
    private static PointF CentreOfPoints(PointF[] point)
    {
        PointF pointFmin = new(int.MaxValue, int.MaxValue);
        PointF pointFmax = new(-1, -1);

        foreach (PointF p in point)
        {
            if (p.X < pointFmin.X) pointFmin = new PointF(p.X, pointFmin.Y);
            if (p.Y < pointFmin.Y) pointFmin = new PointF(pointFmin.X, p.Y);
            if (p.X > pointFmax.X) pointFmax = new PointF(p.X, pointFmax.Y);
            if (p.Y > pointFmax.Y) pointFmax = new PointF(pointFmax.X, p.Y);
        }

        return new PointF((pointFmin.X + pointFmax.X) / 2, (pointFmin.Y + pointFmax.Y) / 2);
    }

    /// <summary>
    /// Draws the region of the sweep that the target is in.
    /// +---++---+
    ///  \  ||  /
    ///   \ || /     hopefully the center strip
    ///    \||/
    ///     \/
    ///     ABM
    /// </summary>
    /// <param name="g"></param>
    internal void DrawWhereTargetIsInRespectToSweepOfHeatSensor(Graphics g, SolidBrush sbColor)
    {
        using Pen pen = new(Color.FromArgb(60, 100, 100, 100));
        pen.DashStyle = DashStyle.Dot;

        // draw the heat sensor
        foreach (PointF[] point in wallSensorTriangleTargetIsInPolygonsInDeviceCoordinates)
        {
            g.FillPolygon(sbColor, point);
            g.DrawPolygon(pen, point);
        }
    }
}
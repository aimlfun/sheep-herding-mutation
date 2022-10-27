using SheepHerderMutation.Configuration;
using SheepHerderMutation.Sheepies;
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
///  ____  _                       ____                            
/// / ___|| |__   ___  ___ _ __   / ___|  ___ _ __  ___  ___  _ __ 
/// \___ \| '_ \ / _ \/ _ \ '_ \  \___ \ / _ \ '_ \/ __|/ _ \| '__|
///  ___) | | | |  __/  __/ |_) |  ___) |  __/ | | \__ \ (_) | |   
/// |____/|_| |_|\___|\___| .__/  |____/ \___|_| |_|___/\___/|_|   
///                       |_|                                      
/// </summary>
internal class SheepSensor
{
    /// <summary>
    /// Stores the sweep triangles, for the visualisation of the sensor. Each PointF[] is a triangle
    /// </summary>
    private readonly List<PointF[]> sheepSensorSweepTrianglePolygonsInDeviceCoordinates = new();

    /// <summary>
    /// Stores the triangle in which we found sheep. Each PointF[] is a triangle.
    /// </summary>
    private readonly List<PointF[]> sheepSensorTriangleTargetIsInPolygonsInDeviceCoordinates = new();

    /// <summary>
    /// Stores where we write the sweep triangle segment labels (numbering).
    /// </summary>
    private readonly List<PointF> sheepSensorTriangleTargetNumberLabelsInDeviceCoordinates = new();

    /// <summary>
    /// Used to track how many sheep are in the triangle relative to complete flock size.
    /// Think of it like percentage/100
    /// </summary>
    private readonly List<float> amountPerTargetTriangle = new();

    /// <summary>
    /// Detects how many sheep in each direction.
    /// </summary>
    /// <param name="angleLifeFormIsPointing"></param>
    /// <param name="predatorLocation"></param>
    /// <param name="sheepLocations"></param>
    /// <param name="sheepSensorRegionsOutput"></param>
    /// <returns></returns>
    internal double[] Read(double angleLifeFormIsPointing, PointF predatorLocation, List<Sheep> sheepLocations, out double[] sheepSensorRegionsOutput)
    {
        sheepSensorTriangleTargetIsInPolygonsInDeviceCoordinates.Clear();
        sheepSensorSweepTrianglePolygonsInDeviceCoordinates.Clear();
        sheepSensorTriangleTargetNumberLabelsInDeviceCoordinates.Clear();
        amountPerTargetTriangle.Clear();

        int SamplePoints = (int)(360F / Config.DogSensorOfSheepVisionAngleInDegrees);

        sheepSensorRegionsOutput = new double[SamplePoints];

        // e.g 
        // input to the neural network
        //   _ \ | / _   
        //   0 1 2 3 4 
        //        
        double fieldOfVisionStartInDegrees = 0;

        //   _ \ | / _   
        //   0 1 2 3 4
        //   [-] this
        double sensorVisionAngleInDegrees = Config.DogSensorOfSheepVisionAngleInDegrees;

        //   _ \ | / _   
        //   0 1 2 3 4
        //   ^ this
        double sensorAngleToCheckInDegrees = fieldOfVisionStartInDegrees - sensorVisionAngleInDegrees / 2 + angleLifeFormIsPointing;

        double DepthOfVisionInPixels = Config.DogSensorOfSheepVisionDepthOfVisionInPixels;

        for (int LIDARangleIndex = 0; LIDARangleIndex < SamplePoints; LIDARangleIndex++)
        {
            //     -45  0  45
            //  -90 _ \ | / _ 90   <-- relative to direction of predator, hence + angle predator is pointing
            double LIDARangleToCheckInRadiansMin = MathUtils.DegreesInRadians(sensorAngleToCheckInDegrees);
            double LIDARangleToCheckInRadiansMax = LIDARangleToCheckInRadiansMin + MathUtils.DegreesInRadians(sensorVisionAngleInDegrees);

            /*  p1        p2
             *   +--------+
             *    \      /
             *     \    /     this is our imaginary "sensor" triangle
             *      \  /
             *       \/
             *    location
             */
            PointF p1 = new((float)(Math.Cos(LIDARangleToCheckInRadiansMin) * DepthOfVisionInPixels + predatorLocation.X),
                            (float)(Math.Sin(LIDARangleToCheckInRadiansMin) * DepthOfVisionInPixels + predatorLocation.Y));

            PointF p2 = new((float)(Math.Cos(LIDARangleToCheckInRadiansMax) * DepthOfVisionInPixels + predatorLocation.X),
                            (float)(Math.Sin(LIDARangleToCheckInRadiansMax) * DepthOfVisionInPixels + predatorLocation.Y));

            sheepSensorTriangleTargetNumberLabelsInDeviceCoordinates.Add(new PointF((p1.X + p2.X) / 2, (p1.Y + p2.Y) / 2));
            sheepSensorSweepTrianglePolygonsInDeviceCoordinates.Add(new PointF[] { predatorLocation, p1, p2 });

            sheepSensorRegionsOutput[LIDARangleIndex] = 0; // no target in this direction

            foreach (Sheep sheep in sheepLocations)
            {
                if (MathUtils.PtInTriangle(sheep.Position, predatorLocation, p1, p2))
                {
#pragma warning disable CS0162 // Unreachable code detected - config drives which code is used, this is not an error.
                    if (Config.AISheepSensorOutputIsDistance)
                    {
                        // plug in distance
                        double dist = MathUtils.DistanceBetweenTwoPoints(sheep.Position, predatorLocation) / DepthOfVisionInPixels;
                        if (sheepSensorRegionsOutput[LIDARangleIndex] == 0 || dist < sheepSensorRegionsOutput[LIDARangleIndex]) sheepSensorRegionsOutput[LIDARangleIndex] = dist;
                    }
                    else
                    {
                        // we're counting sheep, don't fall asleep
                        ++sheepSensorRegionsOutput[LIDARangleIndex];
                    }
#pragma warning restore CS0162 // Unreachable code detected
                }
            }

            if (!Config.AISheepSensorOutputIsDistance) sheepSensorRegionsOutput[LIDARangleIndex] /= sheepLocations.Count;

            // knowing how many sheep doesn't actually help
            if (Config.AIBinarySheepSensor && sheepSensorRegionsOutput[LIDARangleIndex] > 0) sheepSensorRegionsOutput[LIDARangleIndex] = 1;

            if (sheepSensorRegionsOutput[LIDARangleIndex] > 0)
            {
                sheepSensorTriangleTargetIsInPolygonsInDeviceCoordinates.Add(new PointF[] { predatorLocation, p1, p2 });
                amountPerTargetTriangle.Add((float)sheepSensorRegionsOutput[LIDARangleIndex]);
            }

            //   _ \ | / _         _ \ | / _   
            //   0 1 2 3 4         0 1 2 3 4
            //  [-] from this       [-] to this
            sensorAngleToCheckInDegrees += sensorVisionAngleInDegrees;
        }

        return sheepSensorRegionsOutput;
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
        using Pen pen = new(Color.FromArgb(160, 100, 100, 100));

        int segmentLabelIndex = 0;
        PointF[] labels = sheepSensorTriangleTargetNumberLabelsInDeviceCoordinates.ToArray();

        foreach (PointF[] point in sheepSensorSweepTrianglePolygonsInDeviceCoordinates)
        {
            g.FillPolygon(brushOrange, point);
            g.DrawPolygon(pen, point);

            // shows the number of the segment that the sensor (0 is relative to angle of shape)
            if (showSegmentNumber)
            {
                using Font f = new("Arial", 7);

                // centre label on point
                SizeF s = g.MeasureString(segmentLabelIndex.ToString(), f);
                PointF p = new(labels[segmentLabelIndex].X - s.Width / 2,
                                labels[segmentLabelIndex].Y - s.Height / 2);

                g.DrawString(segmentLabelIndex.ToString(), f, Brushes.Black, p);
                ++segmentLabelIndex;
            }
        }
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
    internal void DrawWhereTargetIsInRespectToSweepOfHeatSensor(Graphics g, Color sbColor)
    {
        int amountIndex = 0;

        // draw the heat sensor
        foreach (PointF[] point in sheepSensorTriangleTargetIsInPolygonsInDeviceCoordinates)
        {
            Color color = Color.FromArgb((int)(amountPerTargetTriangle[amountIndex] * 150 + 40).Clamp(0, 255), sbColor.R, sbColor.G, sbColor.B);

            using SolidBrush brush = new(color);

            g.FillPolygon(brush, point);
            ++amountIndex;
        }
    }
}